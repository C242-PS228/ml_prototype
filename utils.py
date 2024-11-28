import numpy as np
import heapq
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model 
from collections import Counter
import stanza
import re
import pandas as pd

emoji_dict = {'😀': 'senyum',
 '😃': 'senyum',
 '😄': 'senang',
 '😁': 'senang',
 '😆': 'senang',
 '😅': 'gugup',
 '🤣': 'tertawa',
 '😂': 'tertawa',
 '🙂': 'senyum',
 '🙃': 'senyum terbalik',
 '\U0001fae0': 'meleleh',
 '😉': 'mengedip',
 '😊': 'senyum',
 '😇': 'senyum',
 '🥰': 'senyum cinta',
 '😍': 'senyum cinta',
 '🤩': 'senyum bintang',
 '😘': 'cium',
 '😗': 'cium',
 '☺': 'senyum',
 '😚': 'cium',
 '😙': 'cium',
 '🥲': 'senyum menangis',
 '😋': 'nikmat',
 '😛': 'mengejek',
 '😜': 'mengejek',
 '🤪': 'mengejek',
 '😝': 'mengejek',
 '🤑': 'uang',
 '🤗': 'pelukan',
 '🤭': 'tertawa',
 '\U0001fae2': 'kaget',
 '\U0001fae3': 'malu',
 '🤫': 'diam kau',
 '🤔': 'berpikir',
 '\U0001fae1': 'hormat',
 '🤐': 'diam',
 '🤨': 'heran',
 '😐': 'netral',
 '😑': 'kesal',
 '\U0001fae5': 'hilang',
 '😏': 'menyeringai',
 '😒': 'tidak senang',
 '🙄': 'kesal',
 '😬': 'meringis',
 '😮\u200d💨': 'menghela napas',
 '🤥': 'bohong',
 '\U0001fae8': 'gemetar',
 '🙂\u200d↔️': 'menggeleng',
 '🙂\u200d↕️': 'mengangguk',
 '😌': 'lega',
 '😔': 'termenung',
 '😪': 'mengantuk',
 '🤤': 'ngiler',
 '😴': 'tidur',
 '😷': 'bau dzaky',
 '🤒': 'sakit',
 '🤕': 'sakit otak',
 '🤢': 'muntah',
 '🤮': 'muntah',
 '🤧': 'sedih',
 '🥵': 'panas',
 '🥶': 'dingin',
 '🥴': 'pusing',
 '😵': 'pingsan',
 '😵\u200d💫': 'bingung',
 '🤯': 'diluar nalar',
 '🤠': 'senang',
 '🥳': 'pesta',
 '😎': 'keren',
 '🧐': 'berpikir',
 '😕': 'bingung',
 '\U0001fae4': 'bingung',
 '😟': 'khawatir',
 '🙁': 'sedih',
 '☹': 'kecewa',
 '😮': 'kaget',
 '😯': 'terpukau',
 '😲': 'terpukau',
 '😳': 'malu',
 '🥺': 'memohon',
 '\U0001f979': 'menahan sedih',
 '😦': 'kaget',
 '😧': 'kaget',
 '😨': 'terkejut',
 '😰': 'cemas',
 '😥': 'sedih',
 '😢': 'sedih',
 '😭': 'nangis',
 '😱': 'terkejut',
 '😞': 'kecewa',
 '😩': 'capek',
 '😫': 'capek',
 '🥱': 'ngantuk',
 '😤': 'mendengus',
 '😡': 'marah',
 '😠': 'marah',
 '🤬': 'marah',
 '😈': 'senyum jahat',
 '👿': 'marah',
 '💀': 'lucu',
 '☠': 'lucu',
 '💩': 'tai',
 '🤡': 'konyol',
 '👺': 'marah',
 '👽': 'lucu',
 '😺': 'senyum',
 '😸': 'senang',
 '😹': 'tertawa',
 '😻': 'cinta',
 '😼': 'menyeringai',
 '😽': 'cium',
 '🙀': 'kaget',
 '😿': 'sedih',
 '😾': 'kesal',
 '🙈': 'mengejek',
 '🙉': 'mengejek',
 '🙊': 'mengejek',
 '💌': 'cinta',
 '💘': 'suka',
 '💝': 'suka',
 '💖': 'suka',
 '💗': 'suka',
 '💓': 'suka',
 '💞': 'suka',
 '💕': 'suka',
 '💟': 'suka',
 '❣': 'suka',
 '💔': 'sedih',
 '❤️\u200d🔥': 'suka',
 '❤️\u200d🩹': 'suka',
 '❤': 'suka',
 '\U0001fa77': 'suka',
 '🧡': 'suka',
 '💛': 'suka',
 '💚': 'suka',
 '💙': 'suka',
 '\U0001fa75': 'suka',
 '💜': 'suka',
 '🤎': 'suka',
 '🖤': 'suka',
 '\U0001fa76': 'suka',
 '🤍': 'suka',
 '💋': 'cium',
 '💯': 'sempurna',
 '💢': 'marah',
 '💥': 'tabrak',
 '💫': 'pusing',
 '💨': 'kentut',
 '💤': 'tidur',
 '👋': 'salam',
 '👌': 'bagus',
 '🤌': 'greget',
 '\U0001faf0': 'cinta',
 '🤟': 'keren',
 '🤘': 'keren',
 '🤙': 'keren',
 '👈': 'menunjuk',
 '👉': 'menunjuk',
 '👆': 'menunjuk',
 '🖕': 'jelek',
 '👇': 'menunjuk',
 '☝': 'menunjuk',
 '\U0001faf5': 'menunjuk',
 '👍': 'bagus',
 '👎': 'jelek',
 '✊': 'semangat',
 '👊': 'memukul',
 '🤛': 'memukul',
 '🤜': 'memukul',
 '👏': 'bagus',
 '🙌': 'apresiasi',
 '\U0001faf6': 'cinta',
 '🤲': 'berdoa',
 '🤝': 'jabat tangan',
 '🙏': 'sopan',
 '✍': 'menulis',
 '💅': 'keren',
 '👀': 'melirik',
 '👅': 'mengejek',
 '👄': 'mulut',
 '🙅': 'tidak',
 '🙅\u200d♂️': 'tidak',
 '🙅\u200d♀️': 'tidak',
 '🧏': 'mewing',
 '🧏\u200d♂️': 'mewing',
 '🧏\u200d♀️': 'mewing',
 '🙇': 'memohon',
 '🙇\u200d♂️': 'memohon',
 '🙇\u200d♀️': 'memohon',
 '🤦': 'kecewa',
 '🤦\u200d♂️': 'kecewa',
 '🤦\u200d♀️': 'kecewa',
 '🤷': 'tidak tau',
 '🤷\u200d♂️': 'tidak tau',
 '🤷\u200d♀️': 'tidak tau',
 '💆': 'pusing',
 '💆\u200d♂️': 'pusing',
 '💆\u200d♀️': 'pusing',
 '🚶': 'jalan',
 '🚶\u200d♂️': 'jalan',
 '🚶\u200d♀️': 'jalan',
 '🚶\u200d➡️': 'jalan',
 '🚶\u200d♀️\u200d➡️': 'jalan',
 '🚶\u200d♂️\u200d➡️': 'jalan',
 '🧍': 'berdiri',
 '🧍\u200d♂️': 'berdiri',
 '🧍\u200d♀️': 'berdiri',
 '🧎': 'berlutut',
 '🧎\u200d♂️': 'berlutut',
 '🧎\u200d♀️': 'berlutut',
 '🧎\u200d➡️': 'berlutut',
 '🧎\u200d♀️\u200d➡️': 'berlutut',
 '🧎\u200d♂️\u200d➡️': 'berlutut',
 '👩\u200d❤️\u200d💋\u200d👨': 'suka',
 '🐒': 'mengejek',
 '🐷': 'mengejek',
 '🐖': 'mengejek',
 '🐽': 'mengejek',
 '🐐': 'ronaldo',
 '🔥': 'keren'}

stop_words = ['dibuat', 'jawab', 'ibaratkan', 'lima', 'adanya', 'berawal', 'bertutur', 'keseluruhan', 'masihkah', 'hanyalah', 'tanyanya', 'sementara', 'bagi', 'meyakini', 'teringat', 'memintakan', 'sekali-kali', 'sebegitu', 'sebutnya', 'dikerjakan', 'pertama', 'sekaligus', 'menegaskan', 'mulanya', 'ingat-ingat', 'semua', 'kami', 'mulai', 'oleh', 'menunjuk', 'maupun', 'bagai', 'ibarat', 'sendirinya', 'bahkan', 'berikan', 'tengah', 'diberikan', 'keduanya', 'sering', 'ditunjukkan', 'berturut-turut', 'beginilah', 'memungkinkan', 'itukah', 'jangankan', 'dipersoalkan', 'nanti', 'sajalah', 'sudahkah', 'tiba', 'pula', 'ucapnya', 'memastikan', 'menunjukkan', 'sangat', 'sesaat', 'keinginan', 'tentunya', 'berlangsung', 'menunjuki', 'apaan', 'sela', 'bisa', 'menanti-nanti', 'apakah', 'semisalnya', 'memisalkan', 'beberapa', 'biasanya', 'bakal', 'disampaikan', 'wong', 'yakni', 'secara', 'semakin', 'akhirnya', 'bahwa', 'tentu', 'terjadinya', 'akhir', 'bagaimanakah', 'dirinya', 'kira', 'sekadarnya', 'kini', 'demi', 'bolehlah', 'lainnya', 'mampu', 'mempersiapkan', 'naik', 'memberi', 'jelaskan', 'akan', 'siap', 'sedang', 'bakalan', 'sebanyak', 'andalah', 'lebih', 'lanjutnya', 'demikianlah', 'semula', 'karena', 'selain', 'kala', 'seusai', 'perlu', 'balik', 'rasa', 'mempersoalkan', 'terkira', 'tunjuk', 'jumlah', 'berapa', 'kalaupun', 'segera', 'kalau', 'diketahuinya', 'sebaik', 'khususnya', 'beginian', 'antar', 'ibaratnya', 'tetap', 'kamulah', 'terlebih', 'boleh', 'menandaskan', 'soal', 'agar', 'lanjut', 'menghendaki', 'namun', 'sebenarnya', 'betul', 'hal', 'sudah', 'begitulah', 'begini', 'kira-kira', 'mendatang', 'memihak', 'mengingat', 'diucapkan', 'bulan', 'makanya', 'meski', 'rasanya', 'hingga', 'punya', 'entahlah', 'sedikit', 'berturut', 'semacam', 'berdatangan', 'dimintai', 'kan', 'lamanya', 'diperlukan', 'sekecil', 'semuanya', 'sama', 'ikut', 'atas', 'kelihatannya', 'dilalui', 'disebutkannya', 'kok', 'tadi', 'misal', 'toh', 'olehnya', 'sangatlah', 'pertama-tama', 'sebagainya', 'diantara', 'keluar', 'mungkinkah', 'sebut', 'dikira', 'bekerja', 'kebetulan', 'diungkapkan', 'terjadilah', 'terakhir', 'jikalau', 'seringnya', 'tanyakan', 'pentingnya', 'waktunya', 'ditandaskan', 'agak', 'bagaikan', 'per', 'betulkah', 'ditunjuk', 'jadilah', 'begitupun', 'dipergunakan', 'inginkah', 'mendatangkan', 'sebutlah', 'dialah', 'bertanya-tanya', 'mau', 'belumlah', 'cara', 'kapan', 'makin', 'semasa', 'menginginkan', 'janganlah', 'sebelumnya', 'ibu', 'mengatakannya', 'sampai', 'diinginkan', 'siapapun', 'berarti', 'belakangan', 'hari', 'akulah', 'bermaksud', 'saja', 'ungkap', 'pertanyaan', 'bapak', 'tuturnya', 'ujar', 'dimaksud', 'dimulailah', 'sebuah', 'seolah', 'dapat', 'percuma', 'kinilah', 'tiga', 'dimaksudkan', 'dijelaskannya', 'mula', 'ada', 'sedemikian', 'tertentu', 'pastilah', 'satu', 'setengah', 'meskipun', 'memerlukan', 'rata', 'macam', 'mempunyai', 'rupanya', 'asalkan', 'atau', 'sebesar', 'sepihak', 'untuk', 'di', 'semisal', 'mengibaratkannya', 'sebisanya', 'tahun', 'yakin', 'dekat', 'dimulainya', 'tandas', 'berlebihan', 'bersama-sama', 'ingin', 'perlukah', 'bung', 'mampukah', 'menyebutkan', 'paling', 'seketika', 'menyeluruh', 'semasih', 'aku', 'disinilah', 'seseorang', 'berkali-kali', 'dahulu', 'ia', 'ditegaskan', 'malahan', 'terdiri', 'antara', 'adapun', 'mengira', 'inginkan', 'sekadar', 'berapakah', 'bukanlah', 'selalu', 'ternyata', 'sudahlah', 'maka', 'mengungkapkan', 'setidaknya', 'entah', 'sehingga', 'memperlihatkan', 'tidakkah', 'diberi', 'tadinya', 'bukan', 'tanya', 'bagaimanapun', 'berakhir', 'berada', 'anda', 'bermacam', 'selanjutnya', 'tahu', 'memberikan', 'dijawab', 'datang', 'setibanya', 'sebegini', 'empat', 'dari', 'secukupnya', 'didatangkan', 'jadi', 'telah', 'sampai-sampai', 'masing', 'dia', 'berlalu', 'tiap', 'mulailah', 'diucapkannya', 'didapat', 'terutama', 'serta', 'jika', 'keadaan', 'se', 'sebagai', 'tambah', 'artinya', 'mengatakan', 'lagian', 'disini', 'mendatangi', 'caranya', 'yaitu', 'kemungkinan', 'menanyai', 'sebaik-baiknya', 'ditujukan', 'turut', 'menanyakan', 'sendirian', 'sebagaimana', 'tandasnya', 'mengetahui', 'tanpa', 'berkenaan', 'melihatnya', 'ini', 'apabila', 'misalkan', 'tersebutlah', 'dimisalkan', 'menjadi', 'sebaliknya', 'sekurangnya', 'menurut', 'ditanyakan', 'mengibaratkan', 'berbagai', 'ditunjuknya', 'tutur', 'menyatakan', 'diibaratkan', 'menanti', 'jumlahnya', 'sesampai', 'berikut', 'setidak-tidaknya', 'mengenai', 'menuju', 'luar', 'seorang', 'demikian', 'sempat', 'dimulai', 'menaiki', 'ataupun', 'bersiap-siap', 'katanya', 'bukannya', 'kata', 'malah', 'meminta', 'tempat', 'berapalah', 'itulah', 'bisakah', 'diperkirakan', 'mana', 'bahwasanya', 'merupakan', 'banyak', 'mempergunakan', 'sama-sama', 'dalam', 'memang', 'sebab', 'segalanya', 'bila', 'bersiap', 'tampaknya', 'serupa', 'sejauh', 'teringat-ingat', 'harusnya', 'sekitar', 'agaknya', 'hendaklah', 'sepantasnyalah', 'hampir', 'melalui', 'terhadapnya', 'sebetulnya', 'tersebut', 'soalnya', 'kelihatan', 'diantaranya', 'mengapa', 'dijelaskan', 'dikatakannya', 'haruslah', 'mengerjakan', 'seingat', 'diperbuat', 'terlihat', 'sebagian', 'semata', 'sekali', 'bolehkah', 'menggunakan', 'sepertinya', 'ketika', 'diperlihatkan', 'tersampaikan', 'nantinya', 'berjumlah', 'berupa', 'minta', 'nah', 'mengucapkan', 'asal', 'diperbuatnya', 'mungkin', 'dikarenakan', 'padanya', 'berikutnya', 'sesuatunya', 'dengan', 'sesudahnya', 'awal', 'begitukah', 'tepat', 'diibaratkannya', 'diri', 'selama-lamanya', 'tentulah', 'dimaksudkannya', 'melainkan', 'pun', 'tidaklah', 'kalaulah', 'mengucapkannya', 'antaranya', 'bukankah', 'ditunjukkannya', 'umumnya', 'kamu', 'diketahui', 'ditunjuki', 'berakhirlah', 'memperbuat', 'pantas', 'tentang', 'menuturkan', 'seberapa', 'kitalah', 'sebelum', 'bermula', 'biasa', 'mendapat', 'pihaknya', 'termasuk', 'gunakan', 'pukul', 'ditambahkan', 'beri', 'kesampaian', 'meyakinkan', 'para', 'inikah', 'sebaiknya', 'bagian', 'sekalian', 'kalian', 'membuat', 'dong', 'tegasnya', 'memulai', 'perlunya', 'dipunyai', 'diingatkan', 'dulu', 'katakan', 'ditanyai', 'jelaslah', 'kiranya', 'terjadi', 'kasus', 'ialah', 'depan', 'sekiranya', 'sewaktu', 'terhadap', 'menambahkan', 'jadinya', 'berapapun', 'segala', 'dituturkannya', 'pihak', 'sampaikan', 'sebabnya', 'melakukan', 'amatlah', 'pasti', 'dibuatnya', 'siapakah', 'tertuju', 'dilakukan', 'kenapa', 'setempat', 'usai', 'terlalu', 'dua', 'setelah', 'sendiri', 'diperlukannya', 'karenanya', 'menyampaikan', 'wah', 'nyaris', 'terbanyak', 'akhiri', 'benar', 'menjawab', 'bawah', 'siapa', 'kamilah', 'masa', 'kita', 'sedangkan', 'seharusnya', 'sinilah', 'ataukah', 'setiba', 'lain', 'ucap', 'masing-masing', 'ungkapnya', 'bersama', 'sejak', 'sana', 'waktu', 'dipastikan', 'tapi', 'wahai', 'waduh', 'ingat', 'diminta', 'menjelaskan', 'setinggi', 'beginikah', 'buat', 'saatnya', 'bermacam-macam', 'dini', 'awalnya', 'dilihat', 'setiap', 'kepadanya', 'terdahulu', 'kelima', 'masih', 'diingat', 'suatu', 'selamanya', 'lah', 'manalagi', 'tampak', 'sekalipun', 'tetapi', 'apatah', 'jawabnya', 'memperkirakan', 'harus', 'pertanyakan', 'itu', 'terdapat', 'saling', 'diakhirinya', 'kembali', 'nyatanya', 'seperlunya', 'tambahnya', 'dan', 'dipertanyakan', 'kapanpun', 'sesegera', 'mengingatkan', 'lagi', 'disebut', 'sambil', 'katakanlah', 'justru', 'sepantasnya', 'keseluruhannya', 'menyiapkan', 'tiba-tiba', 'sini', 'saat', 'kepada', 'manakala', 'berkata', 'selama', 'disebutkan', 'kapankah', 'hendaknya', 'kemudian', 'seluruhnya', 'bagaimana', 'sepanjang', 'padahal', 'melihat', 'walaupun', 'semata-mata', 'seluruh', 'pernah', 'amat', 'diberikannya', 'begitu', 'tegas', 'terasa', 'inilah', 'hanya', 'pada', 'benarlah', 'berakhirnya', 'cuma', 'jelasnya', 'dikatakan', 'sedikitnya', 'seterusnya', 'mendapatkan', 'mengakhiri', 'sesekali', 'sejumlah', 'bilakah', 'masalahnya', 'jawaban', 'sesuatu', 'sesudah', 'misalnya', 'apalagi', 'bertanya', 'terus', 'digunakan', 'selaku', 'menantikan', 'sekarang', 'sayalah', 'mereka', 'pak', 'menanya', 'apa', 'lalu', 'berkehendak', 'juga', 'kemungkinannya', 'ujarnya', 'masalah', 'adalah', 'akankah', 'ditanya', 'sekurang-kurangnya', 'supaya', 'saya', 'seperti', 'yang', 'semampu', 'berkeinginan', 'sekitarnya', 'daripada', 'hendak', 'merekalah', 'ke', 'berujar', 'merasa', 'dimaksudnya', 'semampunya', 'walau', 'usah', 'baru', 'kedua', 'lewat', 'diakhiri', 'sejenak', 'dimungkinkan', 'dituturkan', 'berlainan', 'persoalan', 'menunjuknya', 'menyangkut', 'belakang', 'sesama', 'mempertanyakan', 'benarkah', 'seolah-olah', 'jangan']

nouns = [
    "taste", "flavor", "portion", "service", "price","staff", "menu", "cheese", "topping", "crust", "quality", 
    "size", "material", "design", "style", "customer service", "delivery", "battery", "support", "product", "experience", "complaint", "order", "shipping", "response", "issue", 'cs',
    'ga', 'ngga', 'nggak'
]

exclude_words = ['gua', 'kak', 'gue', 'sih', 'kasih', 'banget', 'orang', 'bu', 'sumpah', 'gitu', 'bnyak', 'banyak', 'gt', 'gitu', 'duo', 'dua', 'satu', 'min', 'pesen', 'brp', 'berapa','memang', 'mmg', 'udh', 'udah', 'uda', 'niat', 'tp', 'tapi']

adjectives = [
    "fresh", "sweet", "spicy", "bland", "cold", 
    "hot", "overpriced", "quick", "trendy", "comfortable", "stylish", "soft", "cheap", "fast", "reliable", "innovative", "responsive", "friendly", "helpful", "unprofessional", 'professional', 
    "amazing", "terrible", "good", "bad", "cozy", "comfy", 'cakep', 'keren', 'gokil'
]


""" DATA PREPROCESSING """

def replace_emoji_with_word(text):
    for emoji, word in emoji_dict.items():
        text = re.sub(f'({emoji})+', f' {word} ', text)
    return text.strip()

def stop_words_removal(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    removed_stop_words = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(removed_stop_words)

def preprocess_text(text):
    text = text.lower()
    text = replace_emoji_with_word(text)
    text = re.sub(r'@\w+', '', text).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stop_words_removal(text)
    return text

# Model and tokenizer
def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def load_nlp_model(path):
    return load_model(path)

def tokenize_batch(texts, tokenizer):
    return tokenizer(
        texts, padding="max_length", truncation=True, max_length=128, return_tensors="tf"
    )['input_ids']

def predict_sentiment_batch(texts, model, tokenizer, preprocess=True):
    if preprocess:
        preprocessed_texts = [preprocess_text(text) for text in texts]
    else:
        preprocessed_texts = texts
    tokenized_texts = tokenize_batch(preprocessed_texts, tokenizer)
    predictions = model.predict(tokenized_texts)
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    sentiments = [sentiment_labels[pred.argmax()] for pred in predictions]
    class_labels = np.argmax(predictions, axis=1)
    return sentiments, class_labels, predictions

""" GETTING INSIGHT """

# TOP 3 POSITIVE AND NEGATIVE COMMENTS
def get_top_3_positive_index(predictions):
    positive_preds = predictions[:, 2]
    pq = []
    for i, pred in enumerate(positive_preds):
        if np.argmax(predictions[i]) == 2:
            heapq.heappush(pq, (-pred, i))
    top_3 = []
    for _ in range(3):
        if pq:
            _, task = heapq.heappop(pq)
            top_3.append(task)
    return top_3

def get_top_3_negative_index(predictions):
    negative_preds = predictions[:, 0]
    pq = []
    for i, pred in enumerate(negative_preds):
        if np.argmax(predictions[i]) == 0:
            heapq.heappush(pq, (-pred, i))
    top_3 = []
    for _ in range(3):
        if pq:
            _, task = heapq.heappop(pq)
            top_3.append(task)
    return top_3

def indices_to_texts(indices, data):
    texts = []
    for index in indices:
        texts.append(data[index])
    return texts

def get_top_3_positive_comments(predictions, data):
    top_3_positive_idx = get_top_3_positive_index(predictions)
    comments = indices_to_texts(top_3_positive_idx, data)

    return comments

def get_top_3_negative_comments(predictions, data):
    top_3_negative_idx = get_top_3_negative_index(predictions)
    comments = indices_to_texts(top_3_negative_idx, data)

    return comments

# MOST COMMON POSITIVE AND NEGATIVE WORDS

def load_stanza_pipeline(lang = 'id', custom_dir = './stanza_models'):
    return stanza.Pipeline(lang, dir=custom_dir)


def analyze_sentiment(preprocessed_texts, class_labels, stanza):   
    nlp = stanza 
    pos_counter = Counter()
    neg_counter = Counter()

    for text, label in zip(preprocessed_texts, class_labels):
        previous_noun = None
        doc = nlp(text)
        
        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text
                
                if word_text in exclude_words:
                    previous_noun = None
                    continue
                
                if word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text
                
                elif (word.upos == "ADJ" or word_text in adjectives) and previous_noun:
                    phrase = f"{previous_noun} {word_text}"
                    if label == 2:
                        pos_counter[phrase] += 1
                    elif label == 0:
                        neg_counter[phrase] += 1
                    previous_noun = None

    pos_common_words = pos_counter.most_common()
    neg_common_words = neg_counter.most_common()
    
    return pos_common_words, neg_common_words

def get_key_words(preprocessed_texts, class_labels, stanza):
    nlp = stanza
    pos_dict = {}
    neg_dict = {}

    for text, label in zip(preprocessed_texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text

                if word_text in exclude_words:
                    previous_noun = None
                    continue

                if word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text

                elif (word.upos == "ADJ" or word_text in adjectives) and previous_noun:
                    phrase = f"{previous_noun} {word_text}"
                    if label == 2:
                        pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                    elif label == 0:
                        neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                    previous_noun = None

    return pos_dict, neg_dict

def get_key_words_and_clean_up(preprocessed_texts, class_labels, stanza, tokenizer, model):
    nlp = stanza
    pos_dict = {}
    neg_dict = {}

    for text, label in zip(preprocessed_texts, class_labels):
        previous_noun = None
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text

                if word_text in exclude_words:
                    previous_noun = None
                    continue

                if word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text

                elif (word.upos == "ADJ" or word_text in adjectives) and previous_noun:
                    phrase = f"{previous_noun} {word_text}"
                    if label == 2:
                        pos_dict[phrase] = pos_dict.get(phrase, 0) + 1
                    elif label == 0:
                        neg_dict[phrase] = neg_dict.get(phrase, 0) + 1
                    previous_noun = None

    pos_arr, neg_arr = get_tag_words(pos_dict, neg_dict)
    if len(pos_arr) > 0:
        pos_tokenized = tokenize_batch(pos_arr, tokenizer)
        true_pos_label = model.predict(pos_tokenized)
        class_labels_pos = np.argmax(true_pos_label, axis=1)

        for i, label in enumerate(class_labels_pos):
            if label != 2:
                word = pos_arr[i]
                del pos_dict[word]

    if len(neg_arr) > 0:
        neg_tokenized = tokenize_batch(neg_arr, tokenizer)
        true_neg_label = model.predict(neg_tokenized)
        class_labels_neg = np.argmax(true_neg_label, axis=1)

        for i, label in enumerate(class_labels_neg):
            if label != 0:
                word = neg_arr[i]
                del neg_dict[word]

    return pos_dict, neg_dict


def get_tag_words(pos_common_words, neg_common_words):
    top_3_pos_words = [word for word, _ in pos_common_words.items()]
    top_3_neg_words = [word for word, _ in neg_common_words.items()]
    
    return top_3_pos_words, top_3_neg_words

def get_tag_words_bruh(texts, class_labels, stanza):
    pos_common_words, neg_common_words = analyze_sentiment(texts, class_labels, stanza)
    return get_tag_words(pos_common_words, neg_common_words)

def clean_up_key_words(pos_words, neg_words, tokenizer, model):
    if len(pos_words) > 0:
        pos_tokenized = tokenize_batch(pos_words, tokenizer)
        true_pos_label = model.predict(pos_tokenized)
        class_labels_pos = np.argmax(true_pos_label, axis=1)

        new_pos_words = []
        for i, label in enumerate(class_labels_pos):
            if label == 2:
                new_pos_words.append(pos_words[i])
    else:
        new_pos_words = []
    if len(neg_words) > 0:
        neg_tokenized = tokenize_batch(neg_words, tokenizer)
        true_neg_label = model.predict(neg_tokenized)
        class_labels_neg = np.argmax(true_neg_label, axis=1)

        new_neg_words = []

        for i, label in enumerate(class_labels_neg):
            if label == 0:
                new_neg_words.append(neg_words[i])
    else:
        new_neg_words = []

    return new_pos_words, new_neg_words
    
""" QUESTION MODEL """


def get_netral_data(class_labels, data):
    new_data = []
    # print(class_labels)
    for i, label in enumerate(class_labels):
        if label == 1:
            new_data.append(data[i])
    # print(new_data)
    return new_data

def stop_words_removal_question(text):
    tokenizer = RegexpTokenizer(r'\w+|\?')
    tokens = tokenizer.tokenize(text)
    removed_stop_words = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(removed_stop_words)


def preprocess_text_question(text):
    text = text.lower()
    text = replace_emoji_with_word(text)
    text = re.sub(r'@\w+', '', text).strip()
    text = re.sub(r'\d+', '', text)
    text = text.strip()

    text = stop_words_removal_question(text)
    return text

def predict_question_batch(texts, model, tokenizer, preprocess=True, treshold=0.5):
    if preprocess:
        preprocessed_texts = [preprocess_text_question(text) for text in texts]
    else:
        preprocessed_texts = texts
    # print(preprocessed_texts)
    tokenized_texts = tokenize_batch(preprocessed_texts, tokenizer)
    predictions = model.predict(tokenized_texts)
    # print(predictions)
    question_labels = ["Bukan Pertanyaan", "Pertanyaan"]
    class_labels = []
    is_questions = []
    for pred in predictions:
        if pred >= 0.5:
            class_labels.append(1)
            is_questions.append(question_labels[1])
        else:
            class_labels.append(0)
            is_questions.append(question_labels[0])
    return is_questions, class_labels, predictions

def get_questions(netraL_data, class_labels):
    questions_data = []
    for i, label in enumerate(class_labels):
        if label == 1:
            questions_data.append(netraL_data[i])

    return questions_data



