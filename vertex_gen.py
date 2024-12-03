import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud_credentials\high-office-443111-t7-9d7551922c9f.json"

# vertexai.init(project="813833490723", location="us-central1")

# model = GenerativeModel(
#     "projects/813833490723/locations/us-central1/endpoints/540160375912398848",
# )

vertexai.init(project="813833490723", location="us-central1")

model = GenerativeModel(
    "projects/813833490723/locations/us-central1/endpoints/1039637722085457920",
)

chat = model.start_chat()

generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.8,
    "top_p": 1,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

def multiturn_generate_content(prompt):
    response = chat.send_message(prompt, generation_config=generation_config, safety_settings=safety_settings)    
    print("Response: ", response.text)

prompt = """
Cakep euy titanium 🔥
HIDUP ITU BERWARNA makanya Yamaha ciptain motor Fazzio Filano biar warna warni dijalan😁🤩🤩🤩
Merah menyala 🔥🔥🔥
Halo Yamaha, saya indent NMAX turbo saat PRJ, sudah 4 bulan gak dikirim, ini motor Real apa ghoib yah min? 😂
Apakah harga cash nya berlaku di seluruh Indonesia ya? Outlet resmi di palembang di daerah mana ya?
Jadi makin bingung milih yang mana.
Ga ada vbelt trip, sama oil trip
Warna ini pake velg bronzee beuuhh👏👏
Ga ad fitur off/on lampu min? 😅
Bukan hanya canggih tapi keren maksimal ❤️❤️
Kak jelek bgt warna doffnya :( bagusan glossy
emang kuat sih ini motor, nanjak kuat, bensin irit, cuma suspensinya aja harus diempukin lagi😄
true broo 🔥🔥🔥
boros 😢
Aih, kece banget ini 🤩
saya kasih tau jangan beli motor ini ,keluhan nya motor ngebul punya saya ahurnya turun mesin ,ganti semua seher
Mantap jiwa 👏
Desain belakang nya jelek dah kyk perindapan
Tapi sayank... Knp mesti indent barangnya 😢😢😢 padahal banyak yg cari
Kudu sabar mijit throtlenya .. tenaga bawahnya galak🔥
Jadi pengen matic ,kalau naik matic berasa jadi lebih romantis
Keren bertenaga 👏 gassspolll trus 🔥
Kemahalan min
Bedanya apa ya
Tangki bensinnya kurang gede
Sayang bgt bagasi nya kecil
ada yang baru nichhhh😍😍😍
Udah ada blm di dealer ya 🤔
r15m new color nya min kapan rilis? udah indent 4bulan blom ada kabar😢
Mau beli tapi duitnya belum cukup mudah mudahan kebeli beberapa bulan lagi ya aamiin
Bagus desain nya,kalo lebih ramping lebih cakep tuh🔥
Kecewa, beli Nmax turbo sudah 4 bulan gak dikirim. Padahal sudah lunas !
ready kah?
Neo Max
Pengen ikutan
Kecewa GK prnah post MX KING
Harga brapa Ka?
mantul gan
Cakepp bngt❤️
Mewah banget nih warna si Fazio, jadi jatuh cinta
Kenapa beli motor cas sulit +kena biaya tambahan
Satu kurangnya.. DESIGNYA aneh
Boleh minta satu dk kk
Boleh ikut dong
Nmax 2024 kapan loucing 😢
Nmax model 2018 di diler masih ada ga mas/ mba?? Infonya
Motor rusak dijual, tidak terima komplain, malah suruh service. Dasar andah, coba buka dulu email, 1 bulan baru dibalas ujung2nya disuruh service
hadeh hari ini mo beli cash motor aerox di susahin sama sales, dibilang kalo cash indent 3 bulan, kalo kredit langsung dateng🤪, sama ada biaya lagi nambah 2jt buat langsung dateng😂
Tolong ya DM nya dibaca dan di respon!!!!!
Ga ada warna green?
"""
multiturn_generate_content(prompt)