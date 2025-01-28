import cv2
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro-vision')

# 抽出対象キーワードリスト
target_keywords = [
    "宿泊施設", "外観", "客室", "露天風呂", "大浴場", "間取り図", "見取り図",
    "食べ物", "料理",
    "風景", "建物", "街並み", "町並み", "世界遺産", "夕景", "夜景", "動物", "植物", "物品",
    "イベント", "祭り", "花火", "クリスマス", "正月", "ウェディング",
    "乗物", "車体", "船体", "機体",
    "人物", "女", "男", "家族", "似顔絵", "シニア", "子供",
    "キャラクター", "ココロちゃん", "わくわく4", "わくわく5",
    "水彩", "ポップ", "ベタ", "線画", "アイコン", "浮世絵", "俯瞰図", "女性向け", "風景画", "表紙", "地図",
    "春", "桜", "夏", "秋", "紅葉", "冬", "雪",
    "国内", "海外",
    "4K", "HD", "地上撮影", "空撮", "360°/VR", "タイムラプス", "動画テンプレート"
]

def extract_keywords_from_image(image_path):
    """画像からキーワードを抽出する関数"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return []
        
        # Geminiに画像とキーワードリストを渡して認識させる
        prompt_text = f"この画像に写っているものを、以下のキーワードの中から選び、半角スペース区切りで抽出してください。キーワード：{', '.join(target_keywords)}。該当するキーワードがない場合は「該当なし」と回答してください。"
        response = model.generate_content([prompt_text, genai.Image(data=cv2.imencode('.jpg', img)[1].tobytes())])

        if response.text:
            extracted_keywords = [keyword.strip() for keyword in response.text.split() if keyword.strip() in target_keywords]
            return extracted_keywords
        else:
            return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []

def main():
    """メイン処理"""
    df = pd.read_csv("uploadsheet.csv")

    for index, row in df.iterrows():
        image_path = row["画像パス"]
        if pd.isna(image_path):
            print(f"Warning: Image path is missing for row {index}. Skipping.")
            continue
        if not os.path.exists(image_path):
             print(f"Warning: Image not found at path {image_path}. Skipping.")
             continue
            
        extracted_keywords = extract_keywords_from_image(image_path)
        df.at[index, "キーワードフラグ"] = " ".join(extracted_keywords) if extracted_keywords else "該当なし"
        print(f"Processed image at {image_path}. Keywords: {' '.join(extracted_keywords) if extracted_keywords else '該当なし'}")

    df.to_csv("uploadsheet.csv", index=False)
    print("CSV file updated successfully.")


if __name__ == "__main__":
    main()
