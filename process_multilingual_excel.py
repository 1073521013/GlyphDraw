import pandas as pd
import json
import os
import argparse
import shutil

# 定义不同语言的 Yes/No 映射
# 也可以在这里自定义 Instruction，如果需要不同语言的指令
LANG_CONFIG = {
    "English": {"yes": "Yes", "no": "No"},
    "Spanish_Mexico": {"yes": "Sí", "no": "No"},
    "Spanish_Spain": {"yes": "Sí", "no": "No"},
    "Hindi": {"yes": "हाँ", "no": "नहीं"},
    "Indonesian": {"yes": "Ya", "no": "Tidak"},
    "Thai": {"yes": "ใช่", "no": "ไม่"},
    "Chinese_Traditional": {"yes": "是", "no": "否"},
    # 默认/中文 Sheet
    "Sheet2": {"yes": "是", "no": "否"},
    "DEFAULT": {"yes": "是", "no": "否"}
}

def get_lang_config(sheet_name):
    # 精确匹配
    if sheet_name in LANG_CONFIG:
        return LANG_CONFIG[sheet_name]
    # 模糊匹配 (比如 sheet 名包含 English)
    for key in LANG_CONFIG:
        if key in sheet_name:
            return LANG_CONFIG[key]
    return LANG_CONFIG["DEFAULT"]

def process_and_sync_excel(input_file, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 读取 Excel
    print(f"Reading {input_file}...")
    try:
        all_sheets = pd.read_excel(input_file, sheet_name=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 3. 标签同步逻辑 (同之前)
    source_sheet_name = 'Sheet2'
    if source_sheet_name not in all_sheets:
        for name, df in all_sheets.items():
            if '是否首要' in df.columns:
                source_sheet_name = name
                break
    
    if source_sheet_name not in all_sheets:
        print("Error: Could not find source labels.")
        return

    print(f"Syncing labels from '{source_sheet_name}'...")
    source_df = all_sheets[source_sheet_name]
    label_values = source_df['是否首要'].values
    
    # 统一 Instruction (目前保持中文，如果需要多语言指令请修改这里)
    instruction = "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    
    updated_sheets = {}

    for sheet_name, df in all_sheets.items():
        print(f"Processing {sheet_name}...")
        
        # --- 获取当前语言的 Yes/No 配置 ---
        lang_conf = get_lang_config(sheet_name)
        yes_str = lang_conf['yes']
        no_str = lang_conf['no']

        # --- A. 同步标签 ---
        if len(df) != len(label_values):
            min_len = min(len(df), len(label_values))
            df = df.iloc[:min_len].copy()
            current_labels = label_values[:min_len]
        else:
            current_labels = label_values

        df['是否首要'] = current_labels
        df = df.fillna('')
        updated_sheets[sheet_name] = df

        # --- B. 生成 JSON ---
        dataset = []
        cols = df.columns.tolist()
        
        # 判断是多语言 Sheet 还是原始 Sheet
        if 'Trans_AppName' in cols:
            col_app = 'Trans_AppName'
            col_title = 'Trans_Title'
            col_content = 'Trans_Content'
            col_summary = 'Trans_Summary'
            is_trans = True
        else:
            col_app = 'AppName'
            col_title = 'Title'
            col_content = 'Content'
            col_summary = '摘要'
            is_trans = False

        if col_app not in cols:
            continue

        for index, row in df.iterrows():
            app = str(row.get(col_app, '')).strip()
            title = str(row.get(col_title, '')).strip()
            content = str(row.get(col_content, '')).strip()
            
            if not app and not title and not content:
                continue
            if app == "[TRANSLATION FAILED]":
                continue

            inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
            
            summary_val = str(row.get(col_summary, '')).strip()
            is_primary_val = str(row.get('是否首要', '')).strip().upper()
            
            output_val = None
            
            # 1. 有摘要 -> 输出摘要
            if summary_val and summary_val != "[TRANSLATION FAILED]":
                # 如果是多语言Sheet，这里已经是翻译过的摘要了(Trans_Summary)
                # 无需再做 Yes/No 转换
                output_val = summary_val
            
            # 2. 无摘要但标记为首要 -> 输出对应语言的 "Yes"
            elif is_primary_val == 'Y':
                output_val = yes_str
                
            # 3. 其他 -> 输出对应语言的 "No"
            else:
                output_val = no_str
            
            if output_val:
                dataset.append({
                    "instruction": instruction,
                    "input": inp,
                    "output": output_val
                })

        # 保存 JSON
        if dataset:
            json_filename = f"{sheet_name}.json"
            out_path = os.path.join(output_dir, json_filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {len(dataset)} items to {json_filename} (Yes={yes_str}, No={no_str})")

    # 5. 写回 Excel
    print("Saving updated Excel file...")
    try:
        # 备份
        if not os.path.exists(input_file + ".bak"):
             shutil.copy2(input_file, input_file + ".bak")
             
        with pd.ExcelWriter(input_file, engine='openpyxl') as writer:
            for sheet_name, df in updated_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Success!")
    except Exception as e:
        print(f"Error saving Excel: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/workspace/majian/vlm_data/input/首要_all.xlsx', help="Excel file path")
    parser.add_argument('--output_dir', default='/mnt/workspace/majian/vlm_data/input/jsons', help="Output directory")
    args = parser.parse_args()

    process_and_sync_excel(args.input, args.output_dir)
