import os
import json
import argparse
import torch
import Levenshtein
import regex as re
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.inference.single.models.utils import extract_clean_model_name, extract_clean_filename

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llms = [
    "/mnt/dataset1/pretrained_fm/Tower-Babel_Babel-9B-Chat",
    "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B",
    "/mnt/dataset1/pretrained_fm/internlm_internlm3-8b-instruct",
]

PROMPT_TEMPLATE = """
Bạn là trợ lý AI chuyên đánh giá chuỗi dự đoán (Predict) và câu trả lời đúng (GT).
**Rules:** - Mỗi mẫu tối đa 1 điểm; nếu GT có nhiều đáp án con, điểm = (số đáp án con khớp) / (tổng đáp án con trong GT).
- Thứ tự không quan trọng; các cách viết khác nhưng cùng ý nghĩa vẫn tính là khớp.
- KHÔNG thêm giải thích, chỉ trả về định dạng:
Score: <đáp án con khớp>/<tổng đáp án con>

**Examples:**
Câu hỏi: Có những loại phương tiện nào trong hình ảnh?
GT: xe máy, xe đạp, ô tô
Predict: xe hơi, xe gắn máy
Score: 2/3
Explain: Vì có 2/3 đáp án khớp với GT ("xe máy" đồng nghĩa với "xe gắn máy", "ô tô" đồng nghĩa với "xe hơi").

Câu hỏi: Những năm nào công ty có lợi nhuận hơn 500 triệu?
GT: năm 2010, năm 2015
Predict: 2010 và 2015
Score: 2/2
Explain: Vì cả hai năm đều có trong GT.

Câu hỏi: Có bao nhiêu người trong bức hình?
GT: 5
Predict: trong hình có 5 người
Score: 1/1
Explain: Câu trả lời đúng với GT, chỉ hơi dài hơn một chút.

Câu hỏi: Nhiệt độ ngày 21/2/2020 được dự bá khoảng bao nhiêu độ C?
GT: 30 độ
Predict: 27 độ
Score: 0/1
Explain: Câu trả lời không khớp với GT.

**Bắt đầu đánh giá:**
Câu hỏi: {question}
GT: {gt}
Predict: {predict}
Score:""".strip()


def parse_score(gen: str) -> float:
    text = gen.split("Score:")[-1].strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)", text)
    if not m:
        return 0.0
    num, den = map(int, m.groups())
    return num/den if den else 0.0


def compute_anls(gt: str, predict: str, threshold: float = 0.5) -> float:
    p = predict.replace('"', '').rstrip('.').lower()
    g = gt.lower()
    score = Levenshtein.ratio(p, g)
    return score if score >= threshold else 0.0


def compute_llm_scores_batch(questions, gts, prs, tokenizer, model):
    prompts = [PROMPT_TEMPLATE.format(question=q, gt=gt, predict=pr) for q, gt, pr in zip(questions, gts, prs)]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=600)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)

    return [parse_score(tokenizer.decode(out_ids[i], skip_special_tokens=True).strip()) for i in range(len(prompts))]


def analyze_by_categories(preds):
    category_scores = {
        "Overall": {"total_anls": 0, "total_accuracy": 0, "total_llm": 0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0, "total_accuracy": 0, "total_llm": 0, "count": 0}),
        "Element": defaultdict(lambda: {"total_anls": 0, "total_accuracy": 0, "total_llm": 0, "count": 0}),
        "Operation": defaultdict(lambda: {"total_anls": 0, "total_accuracy": 0, "total_llm": 0, "count": 0}),
    }
    key_mapping = {"answer_source": "Answer type", "element": "Element", "operation": "Operation"}

    for item in preds:
        if not all(key in item for key in ['answer', 'predict']):
            continue

        gt_answer = item['answer']
        pred_answer = item['predict']

        gt_clean = gt_answer.lower().strip()
        pr_clean = pred_answer.lower().strip().rstrip('.').replace('"', '').rstrip('>').lstrip('<')

        anls_score = compute_anls(gt_clean, pr_clean)
        accuracy_score = int(gt_clean == pr_clean)
        llm_score = item.get('llm_score', 0)

        item.update({'anls': anls_score, 'accuracy': accuracy_score})

        overall = category_scores["Overall"]
        overall["total_anls"] += anls_score
        overall["total_accuracy"] += accuracy_score
        overall["total_llm"] += llm_score
        overall["count"] += 1

        for category_key_orig, category_key_mapped in key_mapping.items():
            value = item.get(category_key_orig)
            values_to_process = (value if isinstance(value, list) and value else
                                 [value] if value is not None else ["N/A"]) 

            for sub_category in values_to_process:
                normalized_key = str(sub_category).lower()
                stats = category_scores[category_key_mapped][normalized_key]
                
                stats["total_anls"] += anls_score
                stats["total_accuracy"] += accuracy_score
                stats["total_llm"] += llm_score
                stats["count"] += 1

    return category_scores


def calculate_averages(stats):
    count = stats['count']
    if count == 0:
        return {'accuracy': 0, 'anls': 0, 'llm_score': 0, 'count': 0}
    return {
        'accuracy': round((stats['total_accuracy'] / count) * 100, 2),
        'anls': round((stats['total_anls'] / count) * 100, 2),
        'llm_score': round((stats['total_llm'] / count) * 100, 2),
    }


def create_detailed_report(category_scores):
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    for category_key, sub_categories in category_scores.items():
        if category_key == "Overall":
            continue
        report[category_key] = {
            sub_category: calculate_averages(stats)
            for sub_category, stats in sorted(sub_categories.items())
        }
    return report


def save_analysis_to_txt(detailed_analysis: dict, filename: str, metric: str):
    """Saves the detailed analysis to a tab-separated .txt file."""
    main_categories_order = ["Answer type", "Element", "Operation"]
    excluded_keys = {'none', '[]', 'n/a'} 
    
    all_sub_categories = defaultdict(set)
    for analysis_data in detailed_analysis.values():
        for category in main_categories_order:
            if category in analysis_data:
                all_sub_categories[category].update(analysis_data[category].keys())

    final_column_structure = {}
    header = ["Method", "Overall"]
    for category in main_categories_order:
        valid_sub_cats = sorted([
            key for key in all_sub_categories[category] if str(key) not in excluded_keys
        ])
        final_column_structure[category] = valid_sub_cats
        header.extend([sub.replace("-", " ").capitalize() for sub in valid_sub_cats])

    all_rows = []
    for model_name, analysis_data in sorted(detailed_analysis.items()):
        row = [model_name]
        row.append(analysis_data.get("Overall", {}).get(metric, 0))
        
        for category in main_categories_order:
            for sub_category in final_column_structure[category]:
                score = analysis_data.get(category, {}).get(sub_category, {}).get(metric, 0)
                row.append(score)
        all_rows.append(row)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\t".join(header) + "\n")
            for row in all_rows:
                f.write("\t".join(map(str, row)) + "\n")
    except IOError as e:
        print(f"Error saving TXT file: {e}")


def load_all_prediction_files():
    if not os.path.exists("results"):
        print("Error: 'results' directory not found.")
        return {}
    predict_files = sorted([f for f in os.listdir("results") if f.endswith(".json") and 'scores' not in f])
    file_data = {}

    for fname in predict_files:
        file_path = os.path.join("results", fname)
        with open(file_path, "r", encoding="utf-8") as f:
            file_data[fname] = json.load(f)

    return file_data


def save_file_data(fname, data):
    file_path = os.path.join("results", fname)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_final_metrics(file_data):
    results = {}
    detailed_analysis = {}

    for fname, preds in file_data.items():
        key = extract_clean_filename(fname)

        # Trung bình từ các *_score nếu có; nếu tắt LLM score, trường này sẽ không tồn tại -> 0.0
        llm_fields = [extract_clean_model_name(model_name) + "_score" for model_name in llms]
        for entry in preds:
            valid_llm_scores = [entry[f] for f in llm_fields if f in entry]
            entry["llm_score"] = sum(valid_llm_scores) / len(valid_llm_scores) if valid_llm_scores else 0.0

        category_scores = analyze_by_categories(preds)
        detailed_analysis[key] = create_detailed_report(category_scores)

        overall_metrics = detailed_analysis[key]['Overall']
        results[key] = {
            "accuracy": overall_metrics['accuracy'],
            "anls": overall_metrics['anls'],
            "llm_score": overall_metrics['llm_score']
        }

        save_file_data(fname, preds)

    return results, detailed_analysis


def run_llm_scoring_for_files(file_data):
    """Chạy tính điểm LLM cho tất cả file chưa có *_score tương ứng."""
    for model_name in llms:
        clean_name = extract_clean_model_name(model_name)
        field = f"{clean_name}_score"

        files_to_process = {fname: data for fname, data in file_data.items()
                            if not all(field in p for p in data)}

        if not files_to_process:
            print(f"✅ {clean_name} scores already exist for all files, skipping...")
            continue

        print(f"🔄 Loading {clean_name} model for {len(files_to_process)} files...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True
        ).eval()

        for fname, preds in files_to_process.items():
            clean_fname = extract_clean_filename(fname)
            print(f"   ⏳ Computing {clean_name} scores for {clean_fname}...")

            batch_size = 32
            n = len(preds)
            all_scores = []

            for start in tqdm(range(0, n, batch_size), desc=f"      {clean_name}"):
                end = min(start + batch_size, n)
                batch_data = preds[start:end]

                questions = [item.get("question", "") for item in batch_data]
                gts = [item.get("answer", "").lower() for item in batch_data]
                prs = [item.get("predict", "").lower().rstrip('.') for item in batch_data]

                scores = compute_llm_scores_batch(questions, gts, prs, tokenizer, model)
                all_scores.extend(scores)

            for i, score in enumerate(all_scores):
                preds[i][field] = score

            save_file_data(fname, preds)
            print(f"   💾 Saved {clean_name} scores for {clean_fname}")

        del model, tokenizer
        torch.cuda.empty_cache()
        print(f"✅ Completed {clean_name} scoring for all files")


def main(llm_scores: bool = False):
    print("🚀 Starting optimized score calculation...")

    file_data = load_all_prediction_files()
    if not file_data:
        print("No prediction files found. Exiting...")
        return
        
    print(f"📁 Loaded {len(file_data)} prediction files")

    if llm_scores:
        print("🧠 LLM scoring: ENABLED")
        run_llm_scoring_for_files(file_data)
    else:
        print("🧠 LLM scoring: DISABLED — sẽ bỏ qua bước load model và sinh *_score")

    print("📈 Computing final metrics and detailed analysis...")
    results, detailed_analysis = compute_final_metrics(file_data)

    print("💾 Saving final results to results folder...")
    os.makedirs("results", exist_ok=True)
    
    with open("results/final_scores.json", "w", encoding="utf-8") as fw:
        json.dump(results, fw, indent=2, ensure_ascii=False)

    with open("results/detailed_analysis.json", "w", encoding="utf-8") as fw:
        json.dump(detailed_analysis, fw, indent=2, ensure_ascii=False)

    print("✅ Final Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("📊 Detailed analysis saved to: results/detailed_analysis.json")

    print("\n--- 💾 Saving detailed analysis to TXT files in results folder ---")

    acc_txt_filename = "results/detailed_analysis_accuracy.txt"
    save_analysis_to_txt(detailed_analysis, acc_txt_filename, metric='accuracy')
    print(f"✅ Accuracy scores saved to: {acc_txt_filename}")

    anls_txt_filename = "results/detailed_analysis_anls.txt"
    save_analysis_to_txt(detailed_analysis, anls_txt_filename, metric='anls')
    print(f"✅ ANLS scores saved to: {anls_txt_filename}")

    llm_txt_filename = "results/detailed_analysis_llm_score.txt"
    save_analysis_to_txt(detailed_analysis, llm_txt_filename, metric='llm_score')
    print(f"✅ LLM scores saved to: {llm_txt_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring pipeline with optional LLM rubric scoring")
    parser.add_argument(
        "--llm-scores", dest="llm_scores", action="store_true",
        help="Bật tính điểm LLM rubric (mặc định: bật)"
    )
    parser.add_argument(
        "--no-llm-scores", dest="llm_scores", action="store_false",
        help="Tắt tính điểm LLM rubric (bỏ qua bước load model)"
    )
    parser.set_defaults(llm_scores=False)
    args = parser.parse_args()

    main(llm_scores=args.llm_scores)
