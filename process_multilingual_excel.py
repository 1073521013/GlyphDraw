import pandas as pd
import json
import os
import argparse
import shutil

def process_and_sync_excel(input_file, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. 读取 Excel 文件所有 Sheet
    print(f"Reading {input_file}...")
    try:
        # 读取所有 sheet 为字典 {sheet_name: dataframe}
        all_sheets = pd.read_excel(input_file, sheet_name=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 3. 确定基准标签列
    source_sheet_name = 'Sheet2' # 根据用户描述，Sheet2 是包含标签的源
    if source_sheet_name not in all_sheets:
        # 如果没有 Sheet2，尝试找第一个包含 '是否首要' 的 sheet
        for name, df in all_sheets.items():
            if '是否首要' in df.columns:
                source_sheet_name = name
                break
    
    if source_sheet_name not in all_sheets:
        print("Error: Could not find a source sheet with '是否首要' column.")
        return

    print(f"Using '{source_sheet_name}' as the source for labels.")
    source_df = all_sheets[source_sheet_name]
    # 提取标签列，确保索引对齐（假设所有sheet行序一致）
    # 使用 values 避免索引不一致带来的问题，因为翻译过程通常保持行序
    label_values = source_df['是否首要'].values 
    
    # 统一的中文指令
    instruction = "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"

    updated_sheets = {}

    # 4. 遍历处理每个 Sheet
    for sheet_name, df in all_sheets.items():
        print(f"Processing sheet: {sheet_name}...")
        
        # --- 步骤 A: 同步标签 ---
        # 确保行数匹配（允许最后几行可能有空行差异，截断或填充）
        if len(df) != len(label_values):
            print(f"Warning: Sheet '{sheet_name}' has {len(df)} rows, but source has {len(label_values)}. Labels might not align perfectly.")
            # 简单对齐：取交集长度
            min_len = min(len(df), len(label_values))
            df = df.iloc[:min_len].copy()
            current_labels = label_values[:min_len]
        else:
            current_labels = label_values

        # 将标签列赋值给当前 Sheet (覆盖或新增)
        df['是否首要'] = current_labels
        df = df.fillna('') # 填充空值
        
        updated_sheets[sheet_name] = df # 保存到待写回列表

        # --- 步骤 B: 生成 JSON ---
        dataset = []
        cols = df.columns.tolist()
        
        # 确定 Input 列名
        if 'Trans_AppName' in cols:
            # 多语言 Sheet
            col_app = 'Trans_AppName'
            col_title = 'Trans_Title'
            col_content = 'Trans_Content'
            col_summary = 'Trans_Summary'
        else:
            # 原始中文 Sheet
            col_app = 'AppName'
            col_title = 'Title'
            col_content = 'Content'
            col_summary = '摘要'

        # 检查必要的列是否存在
        if col_app not in cols or col_content not in cols:
            print(f"Skipping JSON generation for {sheet_name}: missing input columns.")
            continue

        for index, row in df.iterrows():
            app = str(row.get(col_app, '')).strip()
            title = str(row.get(col_title, '')).strip()
            content = str(row.get(col_content, '')).strip()
            
            # 过滤无效行
            if not app and not title and not content:
                continue
            if app == "[TRANSLATION FAILED]":
                continue

            inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
            
            # 获取 Output 相关值
            summary_val = str(row.get(col_summary, '')).strip()
            is_primary_val = str(row.get('是否首要', '')).strip().upper()
            
            output_val = None
            
            # --- Output 逻辑 (统一) ---
            # 1. 如果有摘要 -> 输出摘要
            if summary_val and summary_val != "[TRANSLATION FAILED]":
                output_val = summary_val
            # 2. 否则，根据 '是否首要' 标记判断
            elif is_primary_val == 'Y':
                output_val = "是"
            else:
                output_val = "否"
            
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
            print(f"  -> Saved {len(dataset)} items to {json_filename}")
        else:
            print(f"  -> No valid data for {sheet_name}")

    # 5. 写回 Excel (保存同步后的标签)
    print("Saving updated Excel file...")
    # 备份原文件
    backup_file = input_file + ".bak"
    shutil.copy2(input_file, backup_file)
    print(f"Original file backed up to {backup_file}")
    
    try:
        with pd.ExcelWriter(input_file, engine='openpyxl') as writer:
            for sheet_name, df in updated_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Success! Excel file updated with labels.")
    except Exception as e:
        print(f"Error saving Excel: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/workspace/majian/vlm_data/input/首要_all.xlsx', help="Excel file path")
    parser.add_argument('--output_dir', default='/mnt/workspace/majian/vlm_data/input/jsons', help="Output directory for JSONs")
    args = parser.parse_args()

    process_and_sync_excel(args.input, args.output_dir)
