import pandas as pd
import json
import os
import argparse
import shutil
import random

# 定义不同语言的配置：Yes/No 映射 和 Instruction 翻译
LANG_CONFIG = {
    "English": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "Determine if this is a primary notification. If it is a verification code, meal pickup code, or package pickup code, output the summary directly."
    },
    "Spanish_Mexico": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Spanish_Spain": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "Determine si es una notificación principal. Si es un código de verificación, código de recolección de comida o código de paquete, proporcione el resumen directamente."
    },
    "Hindi": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "तय करें कि क्या यह प्राथमिक अधिसूचना है। यदि यह सत्यापन कोड, भोजन पिकअप कोड, या पैकेज पिकअप कोड है, तो सीधे सारांश प्रदान करें।"
    },
    "Indonesian": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "Tentukan apakah ini notifikasi utama. Jika ini adalah kode verifikasi, kode pengambilan makanan, atau kode pengambilan paket, langsung output ringkasannya."
    },
    "Thai": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "ตรวจสอบว่าเป็นการแจ้งเตือนหลักหรือไม่ หากเป็นรหัสยืนยัน รหัสรับอาหาร หรือรหัสรับพัสดุ ให้แสดงสรุปโดยตรง"
    },
    "Chinese_Traditional": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "判斷是否是首要通知，如果驗證碼、取餐碼、取件碼這三類通知，則直接輸出摘要"
    },
    # 默认/中文 Sheet
    "Sheet2": {
        "yes": "Yes", 
        "no": "No",
        "instruction": "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"
    },
    "DEFAULT": {
        "yes": "Yes", 
        "no": "No",
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

def process_and_sync_excel(primary_file, summary_file, output_dir, test_size=200):
    # 1. 创建输出目录结构
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 设置随机种子以保证划分可复现
    random.seed(42)

    # ================= 1. 处理 首要_all.xlsx (包含多语言) =================
    print(f"Reading Primary File: {primary_file}...")
    try:
        primary_sheets = pd.read_excel(primary_file, sheet_name=None)
    except Exception as e:
        print(f"Error reading primary file: {e}")
        return

    # 确定标签来源
    source_sheet_name = 'Sheet2'
    if source_sheet_name not in primary_sheets:
        for name, df in primary_sheets.items():
            if '是否首要' in df.columns:
                source_sheet_name = name
                break
    
    if source_sheet_name not in primary_sheets:
        print("Error: Could not find source labels in primary file.")
        return

    print(f"Syncing labels from '{source_sheet_name}'...")
    source_df = primary_sheets[source_sheet_name]
    label_values = source_df['是否首要'].values
    
    updated_sheets = {}

    for sheet_name, df in primary_sheets.items():
        print(f"Processing Primary Sheet: {sheet_name}...")
        
        lang_conf = get_lang_config(sheet_name)
        yes_str = lang_conf['yes']
        no_str = lang_conf['no']
        instruction_str = lang_conf['instruction']

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
            
            if summary_val and summary_val != "[TRANSLATION FAILED]":
                output_val = summary_val
            elif is_primary_val == 'Y':
                output_val = yes_str
            else:
                output_val = no_str
            
            if output_val:
                dataset.append({
                    "instruction": instruction_str,
                    "input": inp,
                    "output": output_val
                })

        # --- C. 划分训练集和测试集 ---
        save_datasets(dataset, sheet_name, train_dir, test_dir, test_size)

    # 保存首要文件更新
    print("Saving updated Primary Excel file...")
    try:
        if not os.path.exists(primary_file + ".bak"):
             shutil.copy2(primary_file, primary_file + ".bak")
             
        with pd.ExcelWriter(primary_file, engine='openpyxl') as writer:
            for sheet_name, df in updated_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Success!")
    except Exception as e:
        print(f"Error saving Primary Excel: {e}")

    # ================= 2. 处理 摘要.xlsx (纯提取任务) =================
    if os.path.exists(summary_file):
        print(f"\nReading Summary File: {summary_file}...")
        try:
            summary_sheets = pd.read_excel(summary_file, sheet_name=None)
            
            for sheet_name, df in summary_sheets.items():
                print(f"Processing Summary Sheet: {sheet_name}...")
                df = df.fillna('')
                dataset = []
                
                # 默认使用中文 instruction，如果摘要文件也有多语言，需要类似上面的映射逻辑
                # 这里假设摘要文件主要是补充中文提取能力
                lang_conf = get_lang_config(sheet_name) # 尝试获取，如果是默认则是中文
                instruction_str = lang_conf['instruction']

                for index, row in df.iterrows():
                    app = str(row.get('AppName', '')).strip()
                    title = str(row.get('Title', '')).strip()
                    content = str(row.get('Content', '')).strip()
                    summary_val = str(row.get('摘要', '')).strip()

                    if not app and not title and not content:
                        continue
                    
                    # 只有当有摘要内容时才作为样本
                    if summary_val:
                        inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
                        dataset.append({
                            "instruction": instruction_str,
                            "input": inp,
                            "output": summary_val
                        })
                
                # 保存摘要数据集 (文件名增加后缀区分)
                if dataset:
                    save_name = f"Summary_{sheet_name}"
                    save_datasets(dataset, save_name, train_dir, test_dir, test_size)
                    
        except Exception as e:
            print(f"Error reading summary file: {e}")
    else:
        print(f"Summary file not found: {summary_file}")


def save_datasets(dataset, name, train_dir, test_dir, test_size):
    if not dataset:
        print(f"  -> No valid data for {name}")
        return

    random.shuffle(dataset)
    
    if len(dataset) > test_size:
        test_set = dataset[:test_size]
        train_set = dataset[test_size:]
    else:
        print(f"  Warning: Total data ({len(dataset)}) is less than requested test size ({test_size}) for {name}. All data used for test.")
        test_set = dataset
        train_set = []

    test_file = os.path.join(test_dir, f"{name}.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    train_file = os.path.join(train_dir, f"{name}.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    print(f"  -> Processed {len(dataset)} items for {name}: {len(test_set)} Test, {len(train_set)} Train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary_input', default='/mnt/workspace/majian/vlm_data/input/首要_all.xlsx', help="Primary Excel file")
    parser.add_argument('--summary_input', default='/mnt/workspace/majian/vlm_data/input/摘要.xlsx', help="Summary Excel file")
    parser.add_argument('--output_dir', default='/mnt/workspace/majian/vlm_data/input/jsons', help="Output directory")
    parser.add_argument('--test_size', default=200, type=int, help="Number of samples for test set")
    args = parser.parse_args()

    process_and_sync_excel(args.primary_input, args.summary_input, args.output_dir, args.test_size)
