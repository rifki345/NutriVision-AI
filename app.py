from flask import Flask, render_template, request
import os
import requests
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model_makanan = load_model("model/model_makanan.keras")
model_gizi = load_model("model/model_transfer.keras")

# Label klasifikasi
class_labels_makanan = [
    "Ayam Goreng", "Burger", "French Fries", "Gado-Gado", "Ikan Goreng",
    "Mie Goreng", "Nasi Goreng", "Nasi Padang", "Pizza", "Rawon",
    "Rendang", "Sate", "Soto"
]
class_labels_gizi = ["Tinggi Karbohidrat", "Tinggi Lemak", "Tinggi Protein"]
nutrition_recommendation = {
    "Tinggi Karbohidrat": "Padukan dengan lauk berprotein tinggi untuk keseimbangan nutrisi.",
    "Tinggi Lemak": "Kurangi konsumsi makanan berminyak, dan tambahkan sayuran segar.",
    "Tinggi Protein": "Padukan dengan sayuran tinggi serat untuk keseimbangan gizi."
}

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_deepseek_saran(nama_makanan, kandungan_gizi):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "API Key tidak ditemukan."

    prompt = (
    f"Saya telah menganalisis gambar makanan. "
    f"Makanan yang terdeteksi adalah '{nama_makanan}' dengan kandungan gizi '{kandungan_gizi}'. "
    f"Berikan saran kombinasi makanan sehat untuk menyeimbangkan makanan tersebut. "
    f"Tuliskan dalam format yang rapi dan mudah dibaca seperti berikut:\n\n"
    f"Untuk menyeimbangkan makanan Rawon, kamu bisa kombinasikan dengan:\n"
    f"1. Sayuran tinggi serat — contohnya: ...\n"
    f"2. Protein rendah lemak — contohnya: ...\n"
    f"3. Lemak sehat — contohnya: ...\n\n"
    f"Alternatif cara penyajian:\n"
    f"Tambahkan tips sehat seperti membatasi karbohidrat, mengganti santan, dll.\n\n"
    f"Gunakan gaya bahasa seperti pelatih nutrisi. Jangan gunakan simbol seperti tanda bintang atau markdown."
    )


    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body  # pakai json= bukan data=json.dumps()
        )
        response.raise_for_status()
        hasil = response.json()
        return hasil["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Gagal mendapatkan saran AI: {str(e)}"

def predict_nutrition(image_path):
    img_array = prepare_image(image_path)

    food_pred = model_makanan.predict(img_array)
    food_index = np.argmax(food_pred)
    food = class_labels_makanan[food_index]
    food_confidence = float(food_pred[0][food_index]) * 100

    gizi_pred = model_gizi.predict(img_array)
    gizi_index = np.argmax(gizi_pred)
    gizi = class_labels_gizi[gizi_index]
    gizi_confidence = float(gizi_pred[0][gizi_index]) * 100

    saran_umum = nutrition_recommendation.get(gizi, "Perbanyak makanan bergizi seimbang.")
    saran_ai = get_deepseek_saran(food, gizi)

    return food, food_confidence, gizi, gizi_confidence, saran_umum, saran_ai

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            food, food_conf, gizi_label, gizi_conf, rekomendasi, rekomendasi_ai = predict_nutrition(filepath)

            return render_template('results.html',
                       image_path=filepath,
                       food=food,
                       food_confidence=round(food_conf, 2),
                       prediction=gizi_label,
                       gizi_confidence=round(gizi_conf, 2),
                       recommendation=rekomendasi,
                       recommendation_ai=rekomendasi_ai)

    return render_template('index.html')

# Run server
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)