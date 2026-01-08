import pandas as pd
import json
import os
import argparse

def process_excel_to_jsons(input_file, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. 读取 Excel 文件
    print(f"Reading {input_file}...")
    try:
        # 读取 Excel 文件，获取所有 sheet 名称
        xls = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    sheet_names = xls.sheet_names
    print(f"Found sheets: {sheet_names}")

    # 统一的中文指令
    instruction = "判断是否是首要通知，如果验证码、取餐码、取件码这三类通知，则直接输出摘要"

    # 3. 遍历每个 Sheet
    for sheet in sheet_names:
        print(f"Processing sheet: {sheet}...")
        try:
            # 读取特定 sheet，强制将空值转为空字符串
            df = pd.read_excel(input_file, sheet_name=sheet)
            df = df.fillna('')
        except Exception as e:
            print(f"Error reading sheet {sheet}: {e}")
            continue

        dataset = []
        cols = df.columns.tolist()
        
        # ---------------------------------------------------------
        # 逻辑分支 1: 原始中文数据 (通常是 Sheet2 或 Sheet1)
        # 特征：包含 '是否首要' 列
        # ---------------------------------------------------------
        if '是否首要' in cols or ('AppName' in cols and 'Trans_AppName' not in cols):
            # 确定列名映射
            col_app = 'AppName'
            col_title = 'Title'
            col_content = 'Content'
            col_primary = '是否首要' if '是否首要' in cols else None
            col_summary = '摘要' if '摘要' in cols else None
            
            for index, row in df.iterrows():
                app = str(row.get(col_app, '')).strip()
                title = str(row.get(col_title, '')).strip()
                content = str(row.get(col_content, '')).strip()
                
                # 跳过空行
                if not app and not title and not content:
                    continue

                # 构造 Input
                inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
                
                summary_val = str(row.get(col_summary, '')).strip() if col_summary else ""
                is_primary_val = str(row.get(col_primary, '')).strip().upper() if col_primary else ""
                
                output_val = None
                
                # --- Output 逻辑 ---
                # 1. 如果有摘要 -> 输出摘要
                if summary_val:
                    output_val = summary_val
                # 2. 否则，根据 '是否首要' 标记判断
                elif is_primary_val == 'Y':
                    output_val = "是"
                else:
                    # 包括 'N', 空值, 或其他情况
                    output_val = "否"
                
                if output_val:
                    dataset.append({
                        "instruction": instruction,
                        "input": inp,
                        "output": output_val
                    })

        # ---------------------------------------------------------
        # 逻辑分支 2: 多语言翻译数据 (English, Spanish_Mexico, etc.)
        # 特征：包含 'Trans_AppName' 等翻译列
        # ---------------------------------------------------------
        elif 'Trans_AppName' in cols:
            # 确定列名
            col_trans_app = 'Trans_AppName'
            col_trans_title = 'Trans_Title'
            col_trans_content = 'Trans_Content'
            col_trans_summary = 'Trans_Summary'
            
            # 检查是否存在 '是否首要' 列（如果原表透传过来了）
            # 如果没有，我们只能处理有摘要的行（正样本），无法准确生成负样本
            # 假设多语言sheet主要用于训练提取能力
            
            for index, row in df.iterrows():
                # 获取翻译后的输入
                app = str(row.get(col_trans_app, '')).strip()
                title = str(row.get(col_trans_title, '')).strip()
                content = str(row.get(col_trans_content, '')).strip()
                summary = str(row.get(col_trans_summary, '')).strip()
                
                # 过滤翻译失败或为空的行
                if app == "[TRANSLATION FAILED]" or (not app and not title and not content):
                    continue

                inp = f"AppName: {app}\nTitle: {title}\nContent: {content}"
                
                # --- Output 逻辑 ---
                # 对于多语言，我们目前只处理有摘要的情况 (即验证码/取件码/取餐码)
                # 因为 instruction 说 "如果...则直接输出摘要"
                # 如果没有摘要，且我们不知道如何用目标语言说 "否" (除非硬编码)，暂且跳过
                if summary and summary != "[TRANSLATION FAILED]":
                    dataset.append({
                        "instruction": instruction,
                        "input": inp,
                        "output": summary
                    })
                # 如果需要处理负样本，可以在这里扩展，但需要对应的翻译标签

        else:
            print(f"Skipping sheet {sheet}: Columns not recognized {cols}")

        # 4. 保存 JSON
        if dataset:
            json_filename = f"{sheet}.json"
            out_path = os.path.join(output_dir, json_filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(dataset)} items to {out_path}")
        else:
            print(f"No valid data found for sheet: {sheet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/mnt/workspace/majian/vlm_data/input/首要_all.xlsx', help="Excel file path")
    parser.add_argument('--output_dir', default='/mnt/workspace/majian/vlm_data/input/jsons', help="Output directory")
    args = parser.parse_args()

    process_excel_to_jsons(args.input, args.output_dir)
