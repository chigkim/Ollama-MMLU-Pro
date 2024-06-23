import os
import re
import json
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument(
	"--url",
	help="base_url, default=localhost:11434/v1",
	default="http://localhost:11434'/v1",
)
parser.add_argument("--api", help="api key, default=api", default="api")
parser.add_argument("--model", help="Model name, default=llama3", default="llama3")
parser.add_argument("--category", type=str, default="all")
parser.add_argument(
	"--parallel", type=int, default=1, help="Number of parallel requests"
)
parser.add_argument(
	"--verbosity", type=int, help="Verbosity level 0-2, default=1", default=1
)
args = parser.parse_args()
client = OpenAI(base_url=args.url, api_key=args.api)


def get_completion(prompt: str):
	response = client.chat.completions.create(
		model=args.model,
		messages=[
			{
				"role": "system",
				"content": "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.",
			},
			{"role": "user", "content": prompt},
		],
		temperature=0.1,
		max_tokens=4096,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0,
		stop=["Question:"],
	)

	return response.choices[0].message.content


def load_mmlu_pro():
	dataset = load_dataset("TIGER-Lab/MMLU-Pro")
	test_df, val_df = dataset["test"], dataset["validation"]
	test_df = preprocess(test_df)
	val_df = preprocess(val_df)
	return test_df, val_df


def preprocess(test_df):
	res_df = []
	for each in test_df:
		options = []
		for opt in each["options"]:
			if opt == "N/A":
				continue
			options.append(opt)
		each["options"] = options
		res_df.append(each)
	res = {}
	for each in res_df:
		if each["category"] not in res:
			res[each["category"]] = []
		res[each["category"]].append(each)
	return res


def format_example(question, options, cot_content=""):
	if cot_content == "":
		cot_content = "Let think step by step."
	if cot_content.startswith("A: "):
		cot_content = cot_content[3:]
	example = "Question: {}\nOptions: ".format(question)
	choice_map = "ABCDEFGHIJ"
	for i, opt in enumerate(options):
		example += "{}. {}\n".format(choice_map[i], opt)
	if cot_content == "":
		example += "Answer: "
	else:
		example += "Answer: " + cot_content + "\n\n"
	return example


def extract_answer(text):
	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	match = re.search(pattern, text)
	if match:
		return match.group(1)
	else:
		if args.verbosity >= 2:
			print("extraction failed")
		return None


def run_single_question(single_question, cot_examples_dict, exist_result):
	exist = True
	q_id = single_question["question_id"]
	for each in exist_result:
		if (
			q_id == each["question_id"]
			and single_question["question"] == each["question"]
		):
			if args.verbosity >= 2:
				print("already exists, skipping.")
			return each["pred"], each["response"], exist
	exist = False
	category = single_question["category"]
	cot_examples = cot_examples_dict[category]
	question = single_question["question"]
	options = single_question["options"]
	prompt = ""
	for each in cot_examples:
		prompt += format_example(each["question"], each["options"], each["cot_content"])
	prompt += format_example(question, options).strip()
	try:
		response = get_completion(prompt)
		prompt = response
	except Exception as e:
		print("error", e)
		return None, None, exist
	pred = extract_answer(response)
	return pred, response, exist


def update_result(output_res_path, lock):
	category_record = {}
	res = []
	success = False
	while not success:
		try:
			if os.path.exists(output_res_path):
				with lock:
					with open(output_res_path, "r") as fi:
						res = json.load(fi)
						for each in res:
							category = each["category"]
							if category not in category_record:
								category_record[category] = {"corr": 0.0, "wrong": 0.0}
							if not each["pred"]:
								random.seed(12345)
								x = random.randint(0, len(each["options"]) - 1)
								if x == each["answer_index"]:
									category_record[category]["corr"] += 1
								else:
									category_record[category]["wrong"] += 1
							elif each["pred"] == each["answer"]:
								category_record[category]["corr"] += 1
								if args.verbosity == 2:
									print("random hit.")
							else:
								category_record[category]["wrong"] += 1
			success = True
		except Exception as e:
			print("Error", e)
	return res, category_record


def evaluate(subjects):
	test_df, dev_df = load_mmlu_pro()
	if not subjects:
		subjects = list(test_df.keys())
	print("assigned subjects", subjects)
	lock = threading.Lock()
	for subject in subjects:
		print(f"Testing {subject}...")
		test_data = test_df[subject]
		output_res_path = os.path.join(output_dir, subject + "_result.json")
		output_summary_path = os.path.join(output_dir, subject + "_summary.json")
		res, category_record = update_result(output_res_path, lock)

		with ThreadPoolExecutor(max_workers=args.parallel) as executor:
			futures = {
				executor.submit(run_single_question, each, dev_df, res): each
				for each in test_data
			}
			for future in tqdm(
				as_completed(futures), total=len(futures), smoothing=0.0
			):
				each = futures[future]
				label = each["answer"]
				category = subject
				pred, response, exist = future.result()
				if exist:
					continue
				if response is not None:
					res, category_record = update_result(output_res_path, lock)
					if category not in category_record:
						category_record[category] = {"corr": 0.0, "wrong": 0.0}
					each["pred"] = pred
					each["response"] = response
					res.append(each)
					if pred is not None:
						if pred == label:
							category_record[category]["corr"] += 1
						else:
							category_record[category]["wrong"] += 1
					else:
						category_record[category]["wrong"] += 1
					save_res(res, output_res_path, lock)
					if args.verbosity >= 1:
						save_summary(category_record, output_summary_path, lock, report=True)
					else:
						save_summary(category_record, output_summary_path, lock)
					res, category_record = update_result(output_res_path, lock)
		save_res(res, output_res_path, lock)
		save_summary(category_record, output_summary_path, lock, report=True)
		print(f"Finished testing {subject}.")


def save_res(res, output_res_path, lock):
	temp = []
	exist_q_id = []
	for each in res:
		if each["question_id"] not in exist_q_id:
			exist_q_id.append(each["question_id"])
			temp.append(each)
		else:
			continue
	res = temp
	with lock:
		with open(output_res_path, "w") as fo:
			fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path, lock, report=False):
	total_corr = 0.0
	total_wrong = 0.0
	for k, v in category_record.items():
		if k == "total":
			continue
		cat_acc = v["corr"] / (v["corr"] + v["wrong"])
		category_record[k]["acc"] = cat_acc
		total_corr += v["corr"]
		total_wrong += v["wrong"]
	acc = total_corr / (total_corr + total_wrong)
	category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
	if report:
		print(
			f"\nCorrect: {int(total_corr)}/{int(total_corr+total_wrong)}, Score: {acc*100:.2f}%"
		)
	with lock:
		with open(output_summary_path, "w") as fo:
			fo.write(json.dumps(category_record))


if __name__ == "__main__":
	assigned_subject = [args.category] if args.category != "all" else []
	output_dir = "eval_results/" + re.sub(r"\W", "-", args.model)
	os.makedirs(output_dir, exist_ok=True)
	start = time.time()
	evaluate(assigned_subject)
	duration = time.time() - start
	duration_td = timedelta(seconds=duration)
	hours, remainder = divmod(duration_td.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	print(
		f"Finished the benchmark in {hours} hours, {minutes} minutes, {seconds} seconds."
	)
