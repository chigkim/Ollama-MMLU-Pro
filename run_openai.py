import os
import re
import json
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import timedelta
import codecs
import tomllib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	"-c", "--config",
	help="Configuration file. Default=config.toml",
	default="config.toml",
)
args = parser.parse_args()
config = tomllib.load(open(args.config, "rb"))
client = OpenAI(base_url=config["server"]["host"], api_key=config["server"]["api_key"], timeout=config["server"]["timeout"])


def get_completion(prompt):
	response = client.chat.completions.create(
		model=config["server"]["model"],
		messages=prompt,
		temperature=config["inference"]["temperature"],
		max_tokens=config["inference"]["max_tokens"],
		top_p=config["inference"]["top_p"],
		frequency_penalty=0,
		presence_penalty=0,
		stop=["Question:"],
		timeout=config["server"]["timeout"],
	)
	return response.choices[0].message.content.strip()


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
		cot_content = "Let's think step by step."
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
		if config["log"]["verbosity"] >= 2:
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
			if config["log"]["verbosity"] >= 2:
				print("already exists, skipping.")
			return None, None, None, exist
	exist = False
	category = single_question["category"]
	cot_examples = cot_examples_dict[category]
	question = single_question["question"]
	options = single_question["options"]
	prompt = ""
	for each in cot_examples:
		prompt += format_example(each["question"], each["options"], each["cot_content"])
	prompt += format_example(question, options).strip()
	prompt = [
		{
			"role": "system",
			"content": config["inference"]["system_prompt"],
		},
		{"role": "user", "content": prompt},
	]
	try:
		response = get_completion(prompt)
	except Exception as e:
		print("error", e)
		return None, None, None, exist
	pred = extract_answer(response)
	return prompt, response, pred, exist


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
								category_record["random"] = {"corr": 0.0, "wrong": 0.0}
							if not each["pred"]:
								random.seed(12345)
								x = random.randint(0, len(each["options"]) - 1)
								if x == each["answer_index"]:
									category_record[category]["corr"] += 1
									category_record["random"]["corr"] += 1
								else:
									category_record[category]["wrong"] += 1
									category_record["random"]["wrong"] += 1
							elif each["pred"] == each["answer"]:
								category_record[category]["corr"] += 1
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
		start = time.time()
		print(f"Testing {subject}...")
		test_data = test_df[subject]
		output_res_path = os.path.join(output_dir, subject + "_result.json")
		output_summary_path = os.path.join(output_dir, subject + "_summary.json")
		res, category_record = update_result(output_res_path, lock)

		with ThreadPoolExecutor(max_workers=config["test"]["parallel"]) as executor:
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
				prompt, response, pred, exist = future.result()
				if exist:
					continue
				if response is not None:
					res, category_record = update_result(output_res_path, lock)
					if category not in category_record:
						category_record[category] = {"corr": 0.0, "wrong": 0.0}
					if config["log"]["log_prompt"]:
						each["prompt"] = prompt
					each["response"] = response
					each["pred"] = pred
					res.append(each)
					if config["log"]["verbosity"] >= 3:
						log_json = {
							"id": each["question_id"],
							"question": each["question"],
							"response": each["response"],
							"pred": each["pred"],
							"answer": each["answer"],
						}
						print("\n" + json.dumps(log_json, indent="\t"))
					if pred is not None:
						if pred == label:
							category_record[category]["corr"] += 1
						else:
							category_record[category]["wrong"] += 1
					else:
						category_record[category]["wrong"] += 1
					save_res(res, output_res_path, lock)
					if config["log"]["verbosity"] >= 1:
						save_summary(
							category_record, output_summary_path, lock, report=True
						)
					else:
						save_summary(category_record, output_summary_path, lock)
					res, category_record = update_result(output_res_path, lock)
		save_res(res, output_res_path, lock)
		save_summary(category_record, output_summary_path, lock, report=True)
		hours, minutes, seconds = elapsed(start)
		print(
			f"Finished testing {subject} in {hours} hours, {minutes} minutes, {seconds} seconds."
		)


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
			fo.write(json.dumps(res, indent="\t"))

def print_score(label, corr, wrong):
	corr = int(corr)
	wrong = int(wrong)
	total = corr+wrong
	acc = corr/total*100
	print(
		f"{label}, Correct: {corr}/{total}, Score: {acc:.2f}%"
	)


def save_summary(category_record, output_summary_path, lock, report=False):
	total_corr = 0.0
	total_wrong = 0.0
	for k, v in category_record.items():
		if k == "total" or k == "random":
			continue
		cat_acc = v["corr"] / (v["corr"] + v["wrong"])
		category_record[k]["acc"] = cat_acc
		total_corr += v["corr"]
		total_wrong += v["wrong"]
	acc = total_corr / (total_corr + total_wrong)
	category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
	if report:
		print_score("Category Total", total_corr, total_wrong)
		if "random" in category_record:
			random_corr = category_record["random"]["corr"]
			random_wrong = category_record["random"]["wrong"]
			print_score("Category Random", random_corr, random_wrong)
			print_score("Category Random Subtracted", total_corr-random_corr, total_wrong-random_wrong)
	with lock:
		with open(output_summary_path, "w") as fo:
			fo.write(json.dumps(category_record, indent="\t"))


def final_report():
	total_corr = 0.0
	total_wrong = 0.0
	random_corr = 0.0
	random_wrong = 0.0
	files = os.listdir(output_dir)
	files.append("total_summary.json")
	files = [file.replace("_summary.json", "") for file in files if "summary.json" in file]
	table = "| "+" | ".join(files)+" |\n"
	table += re.sub(r"\w", "-", table)
	scores = []
	for file in os.listdir(output_dir):
		if "summary.json" in file:
			res = json.load(open(os.path.join(output_dir, file)))
			cat_corr = res["total"]["corr"]
			total_corr += cat_corr
			cat_wrong = res["total"]["wrong"]
			total_wrong += cat_wrong
			scores.append(cat_corr/(cat_corr+cat_wrong))
			if "random" in res:
				random_corr += res["random"]["corr"]
				random_wrong += res["random"]["wrong"]
	print_score("Combined Total", total_corr, total_wrong)
	if random_corr and random_wrong:
		print_score("Combined Random", random_corr, random_wrong)
		print_score("Combined  Random Subtracted", total_corr-random_corr, total_wrong-random_wrong)
	scores.append(total_corr/(total_corr+total_wrong))
	scores = [f"{score*100:.2f}" for score in scores]
	table += "| "+" | ".join(scores)+" |"
	print(table)


def elapsed(start):
	duration = time.time() - start
	duration_td = timedelta(seconds=duration)
	hours, remainder = divmod(duration_td.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	return hours, minutes, seconds


if __name__ == "__main__":
	output_dir = "eval_results/" + re.sub(r"\W", "-", config["server"]["model"])
	os.makedirs(output_dir, exist_ok=True)
	assigned_subject = config["test"]["category"]
	start = time.time()
	evaluate(assigned_subject)
	final_report()
	hours, minutes, seconds = elapsed(start)
	print(
		f"Finished the benchmark in {hours} hours, {minutes} minutes, {seconds} seconds."
	)
