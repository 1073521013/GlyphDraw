import pandas as pd
import json
import os
import argparse
import shutil

# 定义不同语言的配置：Yes/No 映射 和 Instruction 翻译
LANG_CONFIG = {
    "English": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "Determine if this is a primary notification. If it is a verification code, meal pickup code, or package pickup code, output the summary directly."
    },
    "Spanish_Mexico": {
        "yes": "Sí", 
        "no": "No",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Spanish_Spain": {
        "yes": "Sí", 
        "no": "No",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Hindi": {
        "yes": "हाँ", 
        "no": "नहीं",
        "instruction": "तय करें कि क्या यह प्राथमिक अधिसूचना है। यदि यह सत्यापन कोड, भोजन पिकअप कोड, या पैकेज पिकअप कोड है, तो सीधे सारांश प्रदान करें।"
    },
    "Indonesian": {
        "yes": "Ya", 
        "no": "Tidak",
        "instruction": "Tentukan apakah ini notifikasi utama. Jika ini adalah kode verifikasi, kode pengambilan makanan, atau kode pengambilan paket, langsung output ringkasannya."
    },
    "Thai": {
        "yes": "ใช่", 
        "no": "ไม่",
        "instruction": "ตรวจสอบว่าเป็นการแจ้งเตือนหลักหรือไม่ หากเป็นรหัสยืนยัน รหัสรับอาหาร หรือรหัสรับพัสดุ ให้แสดงสรุปโดยตรง"
    },
    "Chinese_Traditional": {
        "yes": "是", 
        "no": "否",
        "instruction": "判斷是否是首要通知，如果驗證碼、取餐碼、取件碼這三類通知，則直接輸出摘要"
    },
    # 默认/中文 Sheet
    "Sheet2": {
        "yes": "是", 
        "no": "否",
        "instruction": "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    },
    "DEFAULT": {
        "yes": "是", 
        "no": "否",
        "instruction": "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    }
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

    # 3. 标签同步逻辑
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
    
    updated_sheets = {}

    for sheet_name, df in all_sheets.items():
        print(f"Processing {sheet_name}...")
        
        # --- 获取当前语言的配置 ---
        lang_conf = get_lang_config(sheet_name)
        yes_str = lang_conf['yes']
        no_str = lang_conf['no']
        instruction_str = lang_conf['instruction'] # 获取对应语言的 instruction

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
        else:
            col_app = 'AppName'
            col_title = 'Title'
            col_content = 'Content'
            col_summary = '摘要'

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
                output_val = summary_val
            
            # 2. 无摘要但标记为首要 -> 输出对应语言的 "Yes"
            elif is_primary_val == 'Y':
                output_val = yes_str
                
            # 3. 其他 -> 输出对应语言的 "No"
            else:
                output_val = no_str
            
            if output_val:
                dataset.append({
                    "instruction": instruction_str, # 使用对应语言的 instruction
                    "input": inp,
                    "output": output_val
                })

        # 保存 JSON
        if dataset:
            json_filename = f"{sheet_name}.json"
            out_path = os.path.join(output_dir, json_filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved {len(dataset)} items to {json_filename} (Lang={sheet_name})")

    # 5. 写回 Excel
    print("Saving updated Excel file...")
    try:
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
