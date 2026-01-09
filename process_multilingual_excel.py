import pandas as pd
import json
import os
import argparse
import shutil
import random
from collections import defaultdict

# 定义不同语言的配置
LANG_CONFIG = {
    "English": {
        "yes": "1", "no": "0",
        "instruction": "Determine if this is a primary notification. If it is a verification code, meal pickup code, or package pickup code, output the summary directly."
    },
    "Spanish_Mexico": {
        "yes": "1", "no": "0",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Spanish_Spain": {
        "yes": "1", "no": "0",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Hindi": {
        "yes": "1", "no": "0",
        "instruction": "तय करें कि क्या यह प्राथमिक अधिसूचना है। यदि यह सत्यापन कोड, भोजन पिकअप कोड, या पैकेज पिकअप कोड है, तो सीधे सारांश प्रदान करें।"
    },
    "Indonesian": {
        "yes": "1", "no": "0",
        "instruction": "Tentukan apakah ini notifikasi utama. Jika ini adalah kode verifikasi, kode pengambilan makanan, atau kode pengambilan paket, langsung output ringkasannya."
    },
    "Thai": {
        "yes": "1", "no": "0",
        "instruction": "ตรวจสอบว่าเป็นการแจ้งเตือนหลักหรือไม่ หากเป็นรหัสยืนยัน รหัสรับอาหาร หรือรหัสรับพัสดุ ให้แสดงสรุปโดยตรง"
    },
    "Chinese_Traditional": {
        "yes": "1", "no": "0",
        "instruction": "判斷是否是首要通知，如果驗證碼、取餐碼、取件碼這三類通知，則直接輸出摘要"
    },
    "Sheet2": {
        "yes": "1", "no": "0",
        "instruction": "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    },
    "DEFAULT": {
        "yes": "1", "no": "0",
        "instruction": "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    }
}

def get_lang_config(sheet_name):
    if sheet_name in LANG_CONFIG:
        return LANG_CONFIG[sheet_name]
    for key in LANG_CONFIG:
        if key in sheet_name:
            return LANG_CONFIG[key]
    return LANG_CONFIG["DEFAULT"]

def process_and_merge_then_split(primary_file, summary_file, output_dir, test_size=200):
    # 1. 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(test_dir): os.makedirs(test_dir)

    random.seed(42)
    
    # 用于存储所有处理后的数据：Key=数据集名称(Sheet名), Value=数据列表
    all_datasets = defaultdict(list)

    # ================= 1. 读取并处理 首要_all.xlsx =================
    print(f"Reading Primary File: {primary_file}...")
    try:
        primary_sheets = pd.read_excel(primary_file, sheet_name=None)
        
        # 1.1 标签同步逻辑
        source_sheet = 'Sheet2'
        if source_sheet not in primary_sheets:
            for n, d in primary_sheets.items():
                if '是否首要' in d.columns:
                    source_sheet = n
                    break
        
        if source_sheet in primary_sheets:
            print(f"Syncing labels from '{source_sheet}'...")
            label_values = primary_sheets[source_sheet]['是否首要'].values
            
            # 临时存储用于回写 Excel
            updated_sheets = {}

            for sheet_name, df in primary_sheets.items():
                # 同步标签
                if len(df) != len(label_values):
                    min_len = min(len(df), len(label_values))
                    df = df.iloc[:min_len].copy()
                    curr_labels = label_values[:min_len]
                else:
                    curr_labels = label_values
                
                df['是否首要'] = curr_labels
                df = df.fillna('')
                updated_sheets[sheet_name] = df # 准备回写

                # 提取数据
                lang_conf = get_lang_config(sheet_name)
                dataset = extract_data_from_df(df, lang_conf, is_primary_file=True)
                
                if dataset:
                    all_datasets[sheet_name].extend(dataset)

            # 回写首要文件
            save_excel_backup(primary_file, updated_sheets)
        else:
            print("Warning: Could not find '是否首要' column in primary file.")

    except Exception as e:
        print(f"Error processing primary file: {e}")


    # ================= 2. 读取并处理 摘要.xlsx =================
    if os.path.exists(summary_file):
        print(f"\nReading Summary File: {summary_file}...")
        try:
            summary_sheets = pd.read_excel(summary_file, sheet_name=None)
            
            for sheet_name, df in summary_sheets.items():
                df = df.fillna('')
                lang_conf = get_lang_config(sheet_name)
                
                # 提取数据 (摘要文件也是作为正样本补充)
                dataset = extract_data_from_df(df, lang_conf, is_primary_file=False)
                
                if dataset:
                    # 为了区分来源，给摘要文件的Sheet名加个前缀，或者如果想合并到同名Sheet，就去掉前缀
                    # 这里默认加前缀 Summary_
                    key_name = f"Summary_{sheet_name}" 
                    all_datasets[key_name].extend(dataset)
                    
        except Exception as e:
            print(f"Error reading summary file: {e}")
    else:
        print(f"Summary file not found: {summary_file}")

    # ================= 3. 统一拆分 训练集 / 测试集 =================
    print(f"\nSplitting datasets (Test Size: {test_size})...")
    for name, data in all_datasets.items():
        if not data: continue
        
        # 随机打乱
        random.shuffle(data)
        
        # 拆分
        if len(data) > test_size:
            test_set = data[:test_size]
            train_set = data[test_size:]
        else:
            print(f"  [Warning] {name}: Total {len(data)} <= {test_size}, all used for test.")
            test_set = data
            train_set = []
            
        # 保存
        with open(os.path.join(test_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(train_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(train_set, f, ensure_ascii=False, indent=2)
            
        print(f"  -> {name}: {len(test_set)} Test, {len(train_set)} Train")


def extract_data_from_df(df, lang_conf, is_primary_file=True):
    """从 DataFrame 提取 Instruction/Input/Output"""
    dataset = []
    cols = df.columns.tolist()
    yes_str = lang_conf['yes']
    no_str = lang_conf['no']
    instruction_str = lang_conf['instruction']

    # 确定列映射
    if is_primary_file and 'Trans_AppName' in cols:
        col_app, col_title, col_content, col_summary = 'Trans_AppName', 'Trans_Title', 'Trans_Content', 'Trans_Summary'
    else:
        col_app, col_title, col_content, col_summary = 'AppName', 'Title', 'Content', '摘要'

    if col_app not in cols: return []

    for _, row in df.iterrows():
        app = str(row.get(col_app, '')).strip()
        title = str(row.get(col_title, '')).strip()
        content = str(row.get(col_content, '')).strip()
        
        if not app and not title and not content: continue
        if app == "[TRANSLATION FAILED]": continue

        inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
        summary_val = str(row.get(col_summary, '')).strip()
        
        output_val = None
        
        # 1. 优先取摘要 (正样本)
        if summary_val and summary_val != "[TRANSLATION FAILED]":
            output_val = summary_val
        
        # 2. 如果是首要文件，处理分类标签 (正负样本)
        elif is_primary_file:
            is_primary = str(row.get('是否首要', '')).strip().upper()
            if is_primary == 'Y':
                output_val = yes_str
            else:
                output_val = no_str
        
        if output_val:
            dataset.append({
                "instruction": instruction_str,
                "input": inp,
                "output": output_val
            })
    return dataset

def save_excel_backup(filepath, sheets_dict):
    try:
        if not os.path.exists(filepath + ".bak"):
            shutil.copy2(filepath, filepath + ".bak")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for name, df in sheets_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print("Primary Excel file updated successfully.")
    except Exception as e:
        print(f"Error saving Excel backup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary_input', default='/mnt/workspace/majian/vlm_data/input/首要_all.xlsx')
    parser.add_argument('--summary_input', default='/mnt/workspace/majian/vlm_data/input/摘要.xlsx')
    parser.add_argument('--output_dir', default='/mnt/workspace/majian/vlm_data/input/jsons')
    parser.add_argument('--test_size', default=200, type=int)
    args = parser.parse_args()

    process_and_merge_then_split(args.primary_input, args.summary_input, args.output_dir, args.test_size)
