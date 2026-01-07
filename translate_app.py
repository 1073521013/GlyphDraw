from openai import OpenAI
import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# 设置环境变量
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 定义目标语言及其对应的Sheet名称
TARGET_LANGS = {
    "en": "English",
    "es_MX": "Spanish_Mexico",
    "es_ES": "Spanish_Spain",
    "hi": "Hindi",
    "id": "Indonesian",
    "th": "Thai",
    "zh_TW": "Chinese_Traditional"
}

def clean_json_response(response_text):
    """
    清洗模型返回的字符串，剥离思考过程，提取 JSON 部分
    """
    if not response_text:
        return {}

    # 1. 剥离 <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2. 去除 Markdown 代码块
    if "```" in text:
        match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # 3. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. 正则寻找最外层的 JSON 对象 {...}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
            
    return {}

def run_translation(client, app_name, title, content, model_name="Qwen2.5-VL-7B-Instruct"):
    """
    调用模型进行多语言翻译
    """
    prompt = f'''
        ### Role
        You are a professional localization expert and translator. Your task is to translate the given notification text and App Name from **Chinese** into the following specific languages.
        
        ### Target Languages
        1. **English (en)**
        2. **Spanish - Mexico (es_MX)**: Use Latin American vocabulary/grammar.
        3. **Spanish - Spain (es_ES)**: Use Peninsular vocabulary/grammar.
        4. **Hindi (hi)**
        5. **Indonesian (id)**
        6. **Thai (th)**
        7. **Traditional Chinese (zh_TW)**
        
        ### Input Data
        - **AppName**: {app_name}
        - **Title**: {title}
        - **Content**: {content}
        
        ### Requirements
        1. **Accuracy**: Ensure the translation accurately conveys the urgency and meaning of the original "Primary Notification".
        2. **Style**: Maintain the tone of a mobile app push notification (concise, urgent, clear).
        3. **App Name**: Translate or adapt the App Name for the target market. If the App Name is a well-known international brand, keep it in English or its standard local form.
        4. **Format**: Return a strict JSON object where keys are the language codes defined above.
        
        ### Output Format (Strict JSON)
        {{
            "en": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "es_MX": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "es_ES": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "hi": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "id": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "th": {{ "AppName": "...", "Title": "...", "Content": "..." }},
            "zh_TW": {{ "AppName": "...", "Title": "...", "Content": "..." }}
        }}
        Do NOT output any markdown, thinking process, or extra text. Just the JSON object.
        '''

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            extra_body={
                "top_k": 20,
                "temperature": 0.3, # 翻译任务温度调低，保证准确性
                "chat_template_kwargs": {"enable_thinking": False}, # 尝试关闭思考
            },
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "{}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Translation Pipeline", add_help=True)
    parser.add_argument('--port', default="8001", type=str, help="LLM API port")
    parser.add_argument('--input', default="/mnt/workspace/majian/vlm_data/input/首要_all.xlsx", type=str, help="Input Excel file path")
    parser.add_argument('--model', default="Qwen3", type=str, help="Model name to use")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (process only first 3 rows)")
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )

    input_file = args.input

    print(f"Loading data from {input_file}...")
    try:
        # Create dummy data if file doesn't exist for demonstration purposes in this environment
        if not os.path.exists(input_file):
            print(f"File {input_file} not found. Using dummy data for demonstration.")
            # Ensure directory exists if we were to save valid output, but for now just print warning
            # Creating a dummy DF to prevent crash
            df = pd.DataFrame({
                'AppName': ['TestApp', 'WeChat', 'Douyin'],
                'Title': ['Title1', 'Title2', 'Title3'],
                'Content': ['Content1', 'Content2', 'Content3']
            })
        else:
            df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading excel: {e}")
        exit(1)

    # 初始化存储结构：每个语言一个列表
    language_results = {code: [] for code in TARGET_LANGS.keys()}
    
    print(f"Starting translation for {len(df)} rows using model {args.model}...")

    # 遍历所有行
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Translating"):
        if args.debug and index > 2:
            break

        app_name = str(row.get('AppName', ''))
        title = str(row.get('Title', ''))
        content = str(row.get('Content', ''))
        
        # 调用模型翻译
        llm_response = run_translation(client, app_name, title, content, model_name=args.model)
        
        # 解析结果
        translations = clean_json_response(llm_response)
        
        # 将翻译结果分发到各个语言的列表中
        for lang_code in TARGET_LANGS.keys():
            trans_data = translations.get(lang_code, {})
            
            # 构造这一行的数据对象
            row_data = {
                "Ref_Row_ID": index + 2,
                "Original_AppName": app_name,
                "Original_Title": title,
                "Original_Content": content,
                "Trans_AppName": trans_data.get("AppName", ""),  # Added Trans_AppName
                "Trans_Title": trans_data.get("Title", ""),
                "Trans_Content": trans_data.get("Content", "")
            }
            
            # 如果翻译失败（字典为空），标记一下
            if not trans_data:
                row_data["Trans_Title"] = "[TRANSLATION FAILED]"
                row_data["Trans_AppName"] = "[TRANSLATION FAILED]"
            
            language_results[lang_code].append(row_data)

    # 保存结果到多个 Sheet
    print("Saving results to Excel...")
    try:
        # If input file was dummy, save to a new file
        if not os.path.exists(input_file):
             save_path = "translated_output.xlsx"
             mode = 'w'
             if_sheet_exists = None
        else:
             save_path = input_file
             mode = 'a'
             if_sheet_exists = 'replace'

        with pd.ExcelWriter(save_path, mode=mode, engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
            
            for lang_code, sheet_name in TARGET_LANGS.items():
                data_list = language_results[lang_code]
                if data_list:
                    lang_df = pd.DataFrame(data_list)
                    lang_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved sheet: {sheet_name} ({len(lang_df)} rows)")
                    
        print(f"Success! All translations saved to {save_path}")
        
    except Exception as e:
        print(f"Error saving to excel: {e}")
        # 备份方案
        backup_file = "backup_translated.xlsx"
        try:
            with pd.ExcelWriter(backup_file, engine='openpyxl') as writer:
                 for lang_code, sheet_name in TARGET_LANGS.items():
                    data_list = language_results[lang_code]
                    if data_list:
                        pd.DataFrame(data_list).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved to backup file instead: {backup_file}")
        except Exception as backup_e:
            print(f"Critical error saving backup: {backup_e}")
