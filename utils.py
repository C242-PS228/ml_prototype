import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import heapq
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model 
from collections import Counter
import stanza
import re
import pandas as pd
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import random

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

text_to_emoji_dict = {'senyum': '😺',
 'senang': '😸',
 'gugup': '😅',
 'tertawa': '😹',
 'senyum terbalik': '🙃',
 'meleleh': '\U0001fae0',
 'mengedip': '😉',
 'senyum cinta': '😍',
 'senyum bintang': '🤩',
 'cium': '💋',
 'senyum menangis': '🥲',
 'nikmat': '😋',
 'mengejek': '🐽',
 'uang': '🤑',
 'pelukan': '🤗',
 'kaget': '🙀',
 'malu': '😳',
 'diam kau': '🤫',
 'berpikir': '🧐',
 'hormat': '\U0001fae1',
 'diam': '🤐',
 'heran': '🤨',
 'netral': '😐',
 'kesal': '😾',
 'hilang': '\U0001fae5',
 'menyeringai': '😼',
 'tidak senang': '😒',
 'meringis': '😬',
 'menghela napas': '😮\u200d💨',
 'bohong': '🤥',
 'gemetar': '\U0001fae8',
 'menggeleng': '🙂\u200d↔️',
 'mengangguk': '🙂\u200d↕️',
 'lega': '😌',
 'termenung': '😔',
 'mengantuk': '😪',
 'ngiler': '🤤',
 'tidur': '💤',
 'bau dzaky': '😷',
 'sakit': '🤒',
 'sakit otak': '🤕',
 'muntah': '🤮',
 'sedih': '💔',
 'panas': '🥵',
 'dingin': '🥶',
 'pusing': '💆\u200d♀️',
 'pingsan': '😵',
 'bingung': '\U0001fae4',
 'diluar nalar': '🤯',
 'pesta': '🥳',
 'keren': '🔥',
 'khawatir': '😟',
 'kecewa': '🤦\u200d♀️',
 'terpukau': '😲',
 'memohon': '🙇\u200d♀️',
 'menahan sedih': '\U0001f979',
 'terkejut': '😱',
 'cemas': '😰',
 'nangis': '😭',
 'capek': '😫',
 'ngantuk': '🥱',
 'mendengus': '😤',
 'marah': '💢',
 'senyum jahat': '😈',
 'lucu': '👽',
 'tai': '💩',
 'konyol': '🤡',
 'cinta': '\U0001faf6',
 'suka': '👩\u200d❤️\u200d💋\u200d👨',
 'sempurna': '💯',
 'tabrak': '💥',
 'kentut': '💨',
 'salam': '👋',
 'bagus': '👏',
 'greget': '🤌',
 'menunjuk': '\U0001faf5',
 'jelek': '👎',
 'semangat': '✊',
 'memukul': '🤜',
 'apresiasi': '🙌',
 'berdoa': '🤲',
 'jabat tangan': '🤝',
 'sopan': '🙏',
 'menulis': '✍',
 'melirik': '👀',
 'mulut': '👄',
 'tidak': '🙅\u200d♀️',
 'mewing': '🧏\u200d♀️',
 'tidak tau': '🤷\u200d♀️',
 'jalan': '🚶\u200d♂️\u200d➡️',
 'berdiri': '🧍\u200d♀️',
 'berlutut': '🧎\u200d♂️\u200d➡️',
 'ronaldo': '🐐'}

stop_words = {'dibuat', 'jawab', 'ibaratkan', 'lima', 'adanya', 'berawal', 'bertutur', 'keseluruhan', 'masihkah', 'hanyalah', 'tanyanya', 'sementara', 'bagi', 'meyakini', 'teringat', 'memintakan', 'sekali-kali', 'sebegitu', 'sebutnya', 'dikerjakan', 'pertama', 'sekaligus', 'menegaskan', 'mulanya', 'ingat-ingat', 'semua', 'kami', 'mulai', 'oleh', 'menunjuk', 'maupun', 'bagai', 'ibarat', 'sendirinya', 'bahkan', 'berikan', 'tengah', 'diberikan', 'keduanya', 'sering', 'ditunjukkan', 'berturut-turut', 'beginilah', 'memungkinkan', 'itukah', 'jangankan', 'dipersoalkan', 'nanti', 'sajalah', 'sudahkah', 'tiba', 'pula', 'ucapnya', 'memastikan', 'menunjukkan', 'sangat', 'sesaat', 'keinginan', 'tentunya', 'berlangsung', 'menunjuki', 'apaan', 'sela', 'bisa', 'menanti-nanti', 'apakah', 'semisalnya', 'memisalkan', 'beberapa', 'biasanya', 'bakal', 'disampaikan', 'wong', 'yakni', 'secara', 'semakin', 'akhirnya', 'bahwa', 'tentu', 'terjadinya', 'akhir', 'bagaimanakah', 'dirinya', 'kira', 'sekadarnya', 'kini', 'demi', 'bolehlah', 'lainnya', 'mampu', 'mempersiapkan', 'naik', 'memberi', 'jelaskan', 'akan', 'siap', 'sedang', 'bakalan', 'sebanyak', 'andalah', 'lebih', 'lanjutnya', 'demikianlah', 'semula', 'karena', 'selain', 'kala', 'seusai', 'perlu', 'balik', 'rasa', 'mempersoalkan', 'terkira', 'tunjuk', 'jumlah', 'berapa', 'kalaupun', 'segera', 'kalau', 'diketahuinya', 'sebaik', 'khususnya', 'beginian', 'antar', 'ibaratnya', 'tetap', 'kamulah', 'terlebih', 'boleh', 'menandaskan', 'soal', 'agar', 'lanjut', 'menghendaki', 'namun', 'sebenarnya', 'betul', 'hal', 'sudah', 'begitulah', 'begini', 'kira-kira', 'mendatang', 'memihak', 'mengingat', 'diucapkan', 'bulan', 'makanya', 'meski', 'rasanya', 'hingga', 'punya', 'entahlah', 'sedikit', 'berturut', 'semacam', 'berdatangan', 'dimintai', 'kan', 'lamanya', 'diperlukan', 'sekecil', 'semuanya', 'sama', 'ikut', 'atas', 'kelihatannya', 'dilalui', 'disebutkannya', 'kok', 'tadi', 'misal', 'toh', 'olehnya', 'sangatlah', 'pertama-tama', 'sebagainya', 'diantara', 'keluar', 'mungkinkah', 'sebut', 'dikira', 'bekerja', 'kebetulan', 'diungkapkan', 'terjadilah', 'terakhir', 'jikalau', 'seringnya', 'tanyakan', 'pentingnya', 'waktunya', 'ditandaskan', 'agak', 'bagaikan', 'per', 'betulkah', 'ditunjuk', 'jadilah', 'begitupun', 'dipergunakan', 'inginkah', 'mendatangkan', 'sebutlah', 'dialah', 'bertanya-tanya', 'mau', 'belumlah', 'cara', 'kapan', 'makin', 'semasa', 'menginginkan', 'janganlah', 'sebelumnya', 'ibu', 'mengatakannya', 'sampai', 'diinginkan', 'siapapun', 'berarti', 'belakangan', 'hari', 'akulah', 'bermaksud', 'saja', 'ungkap', 'pertanyaan', 'bapak', 'tuturnya', 'ujar', 'dimaksud', 'dimulailah', 'sebuah', 'seolah', 'dapat', 'percuma', 'kinilah', 'tiga', 'dimaksudkan', 'dijelaskannya', 'mula', 'ada', 'sedemikian', 'tertentu', 'pastilah', 'satu', 'setengah', 'meskipun', 'memerlukan', 'rata', 'macam', 'mempunyai', 'rupanya', 'asalkan', 'atau', 'sebesar', 'sepihak', 'untuk', 'di', 'semisal', 'mengibaratkannya', 'sebisanya', 'tahun', 'yakin', 'dekat', 'dimulainya', 'tandas', 'berlebihan', 'bersama-sama', 'ingin', 'perlukah', 'bung', 'mampukah', 'menyebutkan', 'paling', 'seketika', 'menyeluruh', 'semasih', 'aku', 'disinilah', 'seseorang', 'berkali-kali', 'dahulu', 'ia', 'ditegaskan', 'malahan', 'terdiri', 'antara', 'adapun', 'mengira', 'inginkan', 'sekadar', 'berapakah', 'bukanlah', 'selalu', 'ternyata', 'sudahlah', 'maka', 'mengungkapkan', 'setidaknya', 'entah', 'sehingga', 'memperlihatkan', 'tidakkah', 'diberi', 'tadinya', 'bukan', 'tanya', 'bagaimanapun', 'berakhir', 'berada', 'anda', 'bermacam', 'selanjutnya', 'tahu', 'memberikan', 'dijawab', 'datang', 'setibanya', 'sebegini', 'empat', 'dari', 'secukupnya', 'didatangkan', 'jadi', 'telah', 'sampai-sampai', 'masing', 'dia', 'berlalu', 'tiap', 'mulailah', 'diucapkannya', 'didapat', 'terutama', 'serta', 'jika', 'keadaan', 'se', 'sebagai', 'tambah', 'artinya', 'mengatakan', 'lagian', 'disini', 'mendatangi', 'caranya', 'yaitu', 'kemungkinan', 'menanyai', 'sebaik-baiknya', 'ditujukan', 'turut', 'menanyakan', 'sendirian', 'sebagaimana', 'tandasnya', 'mengetahui', 'tanpa', 'berkenaan', 'melihatnya', 'ini', 'apabila', 'misalkan', 'tersebutlah', 'dimisalkan', 'menjadi', 'sebaliknya', 'sekurangnya', 'menurut', 'ditanyakan', 'mengibaratkan', 'berbagai', 'ditunjuknya', 'tutur', 'menyatakan', 'diibaratkan', 'menanti', 'jumlahnya', 'sesampai', 'berikut', 'setidak-tidaknya', 'mengenai', 'menuju', 'luar', 'seorang', 'demikian', 'sempat', 'dimulai', 'menaiki', 'ataupun', 'bersiap-siap', 'katanya', 'bukannya', 'kata', 'malah', 'meminta', 'tempat', 'berapalah', 'itulah', 'bisakah', 'diperkirakan', 'mana', 'bahwasanya', 'merupakan', 'banyak', 'mempergunakan', 'sama-sama', 'dalam', 'memang', 'sebab', 'segalanya', 'bila', 'bersiap', 'tampaknya', 'serupa', 'sejauh', 'teringat-ingat', 'harusnya', 'sekitar', 'agaknya', 'hendaklah', 'sepantasnyalah', 'hampir', 'melalui', 'terhadapnya', 'sebetulnya', 'tersebut', 'soalnya', 'kelihatan', 'diantaranya', 'mengapa', 'dijelaskan', 'dikatakannya', 'haruslah', 'mengerjakan', 'seingat', 'diperbuat', 'terlihat', 'sebagian', 'semata', 'sekali', 'bolehkah', 'menggunakan', 'sepertinya', 'ketika', 'diperlihatkan', 'tersampaikan', 'nantinya', 'berjumlah', 'berupa', 'minta', 'nah', 'mengucapkan', 'asal', 'diperbuatnya', 'mungkin', 'dikarenakan', 'padanya', 'berikutnya', 'sesuatunya', 'dengan', 'sesudahnya', 'awal', 'begitukah', 'tepat', 'diibaratkannya', 'diri', 'selama-lamanya', 'tentulah', 'dimaksudkannya', 'melainkan', 'pun', 'tidaklah', 'kalaulah', 'mengucapkannya', 'antaranya', 'bukankah', 'ditunjukkannya', 'umumnya', 'kamu', 'diketahui', 'ditunjuki', 'berakhirlah', 'memperbuat', 'pantas', 'tentang', 'menuturkan', 'seberapa', 'kitalah', 'sebelum', 'bermula', 'biasa', 'mendapat', 'pihaknya', 'termasuk', 'gunakan', 'pukul', 'ditambahkan', 'beri', 'kesampaian', 'meyakinkan', 'para', 'inikah', 'sebaiknya', 'bagian', 'sekalian', 'kalian', 'membuat', 'dong', 'tegasnya', 'memulai', 'perlunya', 'dipunyai', 'diingatkan', 'dulu', 'katakan', 'ditanyai', 'jelaslah', 'kiranya', 'terjadi', 'kasus', 'ialah', 'depan', 'sekiranya', 'sewaktu', 'terhadap', 'menambahkan', 'jadinya', 'berapapun', 'segala', 'dituturkannya', 'pihak', 'sampaikan', 'sebabnya', 'melakukan', 'amatlah', 'pasti', 'dibuatnya', 'siapakah', 'tertuju', 'dilakukan', 'kenapa', 'setempat', 'usai', 'terlalu', 'dua', 'setelah', 'sendiri', 'diperlukannya', 'karenanya', 'menyampaikan', 'wah', 'nyaris', 'terbanyak', 'akhiri', 'benar', 'menjawab', 'bawah', 'siapa', 'kamilah', 'masa', 'kita', 'sedangkan', 'seharusnya', 'sinilah', 'ataukah', 'setiba', 'lain', 'ucap', 'masing-masing', 'ungkapnya', 'bersama', 'sejak', 'sana', 'waktu', 'dipastikan', 'tapi', 'wahai', 'waduh', 'ingat', 'diminta', 'menjelaskan', 'setinggi', 'beginikah', 'buat', 'saatnya', 'bermacam-macam', 'dini', 'awalnya', 'dilihat', 'setiap', 'kepadanya', 'terdahulu', 'kelima', 'masih', 'diingat', 'suatu', 'selamanya', 'lah', 'manalagi', 'tampak', 'sekalipun', 'tetapi', 'apatah', 'jawabnya', 'memperkirakan', 'harus', 'pertanyakan', 'itu', 'terdapat', 'saling', 'diakhirinya', 'kembali', 'nyatanya', 'seperlunya', 'tambahnya', 'dan', 'dipertanyakan', 'kapanpun', 'sesegera', 'mengingatkan', 'lagi', 'disebut', 'sambil', 'katakanlah', 'justru', 'sepantasnya', 'keseluruhannya', 'menyiapkan', 'tiba-tiba', 'sini', 'saat', 'kepada', 'manakala', 'berkata', 'selama', 'disebutkan', 'kapankah', 'hendaknya', 'kemudian', 'seluruhnya', 'bagaimana', 'sepanjang', 'padahal', 'melihat', 'walaupun', 'semata-mata', 'seluruh', 'pernah', 'amat', 'diberikannya', 'begitu', 'tegas', 'terasa', 'inilah', 'hanya', 'pada', 'benarlah', 'berakhirnya', 'cuma', 'jelasnya', 'dikatakan', 'sedikitnya', 'seterusnya', 'mendapatkan', 'mengakhiri', 'sesekali', 'sejumlah', 'bilakah', 'masalahnya', 'jawaban', 'sesuatu', 'sesudah', 'misalnya', 'apalagi', 'bertanya', 'terus', 'digunakan', 'selaku', 'menantikan', 'sekarang', 'sayalah', 'mereka', 'pak', 'menanya', 'apa', 'lalu', 'berkehendak', 'juga', 'kemungkinannya', 'ujarnya', 'masalah', 'adalah', 'akankah', 'ditanya', 'sekurang-kurangnya', 'supaya', 'saya', 'seperti', 'yang', 'semampu', 'berkeinginan', 'sekitarnya', 'daripada', 'hendak', 'merekalah', 'ke', 'berujar', 'merasa', 'dimaksudnya', 'semampunya', 'walau', 'usah', 'baru', 'kedua', 'lewat', 'diakhiri', 'sejenak', 'dimungkinkan', 'dituturkan', 'berlainan', 'persoalan', 'menunjuknya', 'menyangkut', 'belakang', 'sesama', 'mempertanyakan', 'benarkah', 'seolah-olah', 'jangan'}

nouns = {
    "taste", "flavor", "portion", "service", "price","staff", "menu", "cheese", "topping", "crust", "quality", 
    "size", "material", "design", "style", "customer service", "delivery", "battery", "support", "product", "experience", "complaint", "order", "shipping", "response", "issue", 'cs',
    'ga', 'ngga', 'nggak', 'bumbu', 'quality', 'control'
}

exclude_words = {'gua', 'kak', 'gue', 'sih', 'kasih', 'banget', 'orang', 'bu', 'sumpah', 'gitu', 'bnyak', 'banyak', 'gt', 'gitu', 'duo', 'dua', 'satu', 'min', 'pesen', 'brp', 'berapa','memang', 'mmg', 'udh', 'udah', 'uda', 'niat', 'tp', 'tapi', 'a', 'i', 'u', 'e', 'o', 'emang', 'emg', 'emng', 'bner', 'bnr', 'plis', 'pls', 'gara'}
exclude_nouns = {'beban', 'nakal', 'biar', 'dpt', 'masih', 'msh', 'mash', 'lu', 'kamu', 'kmu', 'lo', 'nih', 'ni'}

adjectives = {
    "fresh", "sweet", "spicy", "bland", "cold", 
    "hot", "overpriced", "quick", "trendy", "comfortable", "stylish", "soft", "cheap", "fast", "reliable", "innovative", "responsive", "friendly", "helpful", "unprofessional", 'professional', 
    "amazing", "terrible", "good", "bad", "cozy", "comfy", 'cakep', 'keren', 'gokil', 'beban', 'nakal', 'quality', 'control', 'bau', 'basi'
}

negations = {'ga', 'gak', 'gada', 'nggak', 'enggak', 'tidak', 'ngga', 'gakk', 'nggk'}

bad_words = {
 'ahole',
 'anal',
 'anal-play',
 'analingus',
 'analplay',
 'androsodomy',
 'anilingus',
 'anjim',
 'anjing',
 'anjir',
 'anjrit',
 'anjrot',
 'anus',
 'arsehole',
 'ashole',
 'asholes',
 'ashu',
 'ass',
 'ass monkey',
 'ass-playauto-eroticism',
 'asses',
 'assface',
 'asshole',
 'assholes',
 'assholez',
 'assholz',
 'asslick',
 'assplay',
 'asswipe',
 'asu',
 'autofellatio',
 'autopederasty',
 'awuk',
 'ayir',
 'azzhole',
 'babi',
 'bacot',
 'badass',
 'bagudung',
 'bajingan',
 'ball-gag',
 'ballgag',
 'banci',
 'bangke',
 'bangor',
 'bangsat',
 'bareback',
 'barebacking',
 'bastard',
 'bastards',
 'bastardz',
 'bdsm',
 'beastilaity',
 'bego',
 'bejad',
 'bejat',
 'bencong',
 'bestiality',
 'biatch',
 'birahi',
 'bitch',
 'bitches',
 'bloon',
 'blow job',
 'blow-job',
 'blowjob',
 'blowjobs',
 'bodat',
 'bogel',
 'bokep',
 'boob',
 'boobies',
 'boobs',
 'borjong',
 'breas',
 'breasts',
 'brengsek',
 'bubs',
 'bugger',
 'buggery',
 'bugil',
 'bukake',
 'bukakke',
 'bull-dyke',
 'bull-dykes',
 'bulldyke',
 'bulldykes',
 'bundir',
 'bungul',
 'bunuh',
 'burik',
 'burit',
 'butt',
 'butt-pirate',
 'butt-plug',
 'butt-plugs',
 'butthole',
 'buttplug',
 'buttplugs',
 'butts',
 'buttwipe',
 'carpet muncher',
 'cawek',
 'cawk',
 'cazzo',
 'cemen',
 'cerita dewasa',
 'cerita hot',
 'cerita panas',
 'chick',
 'chicks',
 'chink',
 'choda',
 'chraa',
 'chudai',
 'chuj',
 'cipa',
 'cipki',
 'cipok',
 'cium',
 'cl1t',
 'clit',
 'clitoris',
 'clits',
 'cnts',
 'cntz',
 'cock',
 'cock-head',
 'cock-sucker',
 'cockhead',
 'cocks',
 'cocksucker',
 'cok',
 'colai',
 'coli',
 'colmek',
 'coprophagy',
 'coprophilia',
 'cornhole',
 'cornholes',
 'corpophilia',
 'corpophilic',
 'crack',
 'crackz',
 'crap',
 'crut',
 'cukimai',
 'cukimay',
 'culun',
 'cum',
 'cumbu',
 'cumming',
 'cumpic',
 'cums',
 'cumshot',
 'cumshots',
 'cunilingus',
 'cunnilingus',
 'cunt',
 'cunts',
 'cuntz',
 'd1ck',
 'damn',
 'dancuk',
 'deepthroat',
 'defecated',
 'defecating',
 'defecation',
 'dego',
 'desnuda',
 'dewasa',
 'dick',
 'dicks',
 'dike',
 'dildo',
 'dirsa',
 'dnwallace',
 'doggystyle',
 'dominatricks',
 'dominatrics',
 'dominatrix',
 'douche',
 'douches',
 'douching',
 'dyke',
 'dykes',
 'dziwka',
 'eewe',
 'ejackulate',
 'ejakulate',
 'ekrem',
 'encuk',
 'enculer',
 'erection',
 'erections',
 'erotic',
 'erotica',
 'ewe',
 'f u c k',
 'f u c k e r',
 'facesit',
 'facesitting',
 'faen',
 'fag',
 'faget',
 'faggit',
 'faggot',
 'fagit',
 'fags',
 'fagz',
 'faig',
 'faigs',
 'fanculo',
 'fart',
 'farted',
 'farting',
 'fcuk',
 'feces',
 'feg',
 'felch',
 'fetish',
 'fetishes',
 'ficken',
 'fisting',
 'flikker',
 'footjob',
 'foreskin',
 'fotze',
 'four some',
 'foursome',
 'fuck',
 'fucker',
 'fuckin',
 'fucking',
 'fucks',
 'fudge packer',
 'fuk',
 'fukah',
 'fuken',
 'fuker',
 'fukin',
 'fukk',
 'fukkah',
 'fukken',
 'fukker',
 'fukkin',
 'fukr',
 'futkretzn',
 'fuxor',
 'g-spot',
 'g1la',
 'gag',
 'gang-bang',
 'gangbang',
 'gauk',
 'gawk',
 'gawu',
 'gay',
 'gayboy',
 'gaygirl',
 'gays',
 'gayz',
 'gei',
 'gembel',
 'genital',
 'genitalia',
 'genitals',
 'gey',
 'gigolo',
 'gila',
 'glory-hole',
 'glory-holes',
 'gloryhole',
 'gloryholes',
 'goblog',
 'goblok',
 'god-damned',
 'gook',
 'groupsex',
 'gspot',
 'guiena',
 'hand-job',
 'handjob',
 'haram',
 'hardcore',
 'heang',
 'hell',
 'helvete',
 'hencet',
 'henceut',
 'hentai',
 'hitler',
 'hoar',
 'hoer',
 'homosexual',
 'honkey',
 'hoor',
 'hor',
 'hore',
 'horny',
 'hot girl',
 'hot video',
 'hubungan intim',
 'idiot',
 'incest',
 'injun',
 'intercourse',
 'interracial',
 'jablai',
 'jablay',
 'jackass',
 'jackoff',
 'jancok',
 'jancuk',
 'jangkik',
 'jap',
 'japs',
 'jebanje',
 'jemb',
 'jembut',
 'jerk-off',
 'jilat',
 'jingan',
 'jiss',
 'jizz',
 'joanne yiokaris',
 'kacuk',
 'kampang',
 'kampret',
 'kanciang',
 'kancut',
 'kanjut',
 'kawk',
 'kelamin',
 'kent',
 'keparat',
 'kete',
 'kimak',
 'kirik',
 'kl1t',
 'klentit',
 'klimak',
 'klimax',
 'klitoris',
 'knob',
 'kntl',
 'konthol',
 'kontol',
 'koplok',
 'kuksuger',
 'kunt',
 'kunts',
 'kuntz',
 'kunyuk',
 'kutang',
 'kutis',
 'kwontol',
 'labia',
 'labial',
 'lancap',
 'leec',
 'lesbi',
 'lesbian',
 'lesbians',
 'lesbo',
 'lezzian',
 'lipshits',
 'lipshitz',
 'lolita',
 'lolitas',
 'lonte',
 'lucah',
 'maho',
 'mamhoon',
 'maria ozawa',
 'masochism',
 'masochist',
 'masochistic',
 'masokist',
 'massterbait',
 'masstrbait',
 'masstrbate',
 'masterbaiter',
 'masterbate',
 'masterbates',
 'masturbasi',
 'masturbat',
 'masturbate',
 'masturbation',
 'mat1',
 'matamu',
 'matane',
 'mati',
 'mbut',
 'meki',
 'memek',
 'merd*',
 'mesum',
 'mibun',
 'modar',
 'modyar',
 'mofo',
 'mokad',
 'monkleigh',
 'motha fucker',
 'motha fuker',
 'motha fukkah',
 'motha fukker',
 'mother fucker',
 'mother fukah',
 'mother fuker',
 'mother fukkah',
 'mother fukker',
 'mother-fucker',
 'motherfisher',
 'motherfucker',
 'mouliewop',
 'muff',
 'muie',
 'mujeres',
 'mulkku',
 'muschi',
 'mutha fucker',
 'mutha fukah',
 'mutha fuker',
 'mutha fukkah',
 'mutha fukker',
 'najis',
 'naked',
 'nastt',
 'nazi',
 'nazis',
 'ncuk',
 'ndhasmu',
 'necrophilia',
 'nenen',
 'nepesaurio',
 'ngecrot',
 'ngegay',
 'ngentot',
 'ngewe',
 'ngocok',
 'ngolom',
 'ngulum',
 'nigga',
 'nigger',
 'niggers',
 'nigr',
 'niigr',
 'nipple',
 'nipples',
 'no cd',
 'nocd',
 'nthu',
 'ntut',
 'nude',
 'nudes',
 'nudity',
 'nutsack',
 'nympho',
 'nymphomania',
 'nymphomaniac',
 'onani',
 'orafis',
 'orgasim',
 'orgasm',
 'orgasme',
 'orgasms',
 'orgasum',
 'orgies',
 'orgy',
 'oriface',
 'orifice',
 'orifiss',
 'orospu',
 'packi',
 'packie',
 'packy',
 'paki',
 'pakie',
 'paksa',
 'paky',
 'pantat',
 'pantek',
 'paska',
 'pcun',
 'pecker',
 'pecun',
 'pederast',
 'pederasty',
 'pedju',
 'pedophilia',
 'pedophiliac',
 'pee',
 'peeenus',
 'peeenusss',
 'peeing',
 'peenus',
 'peinus',
 'peju',
 'peli',
 'pemerkosaan',
 'penas',
 'penetration',
 'penetrations',
 'penis',
 'penis-breath',
 'pentil',
 'penus',
 'penuus',
 'pepek',
 'perek',
 'perkosa',
 'perse',
 'pervert',
 'perverted',
 'perverts',
 'pg ishazamuddin',
 'phuc',
 'phuck',
 'phuk',
 'phuker',
 'phukker',
 'piatu',
 'picka',
 'pierdol',
 'pilat',
 'pillu',
 'pimmel',
 'pimpis',
 'piss',
 'pizda',
 'polac',
 'polack',
 'polak',
 'poonani',
 'poontsee',
 'poop',
 'porn',
 'porno',
 'precum',
 'preteen',
 'pric',
 'prick',
 'pricks',
 'prik',
 'pron',
 'prostitute',
 'prostituted',
 'prostitutes',
 'prostituting',
 'puki',
 'pukimak',
 'pula',
 'pule',
 'pups',
 'pus1',
 'puss',
 'pusse',
 'pussee',
 'pussies',
 'pussy',
 'pussylips',
 'pussys',
 'puta',
 'puto',
 'puuke',
 'puuker',
 'qahbeh',
 'qontol',
 'queef',
 'queer',
 'queers',
 'queerz',
 'qweef',
 'qweers',
 'qweerz',
 'qweir',
 'racist',
 'rape',
 'raped',
 'rapes',
 'rapist',
 'rautenberg',
 'recktum',
 'rectum',
 'retard',
 'rimjob',
 'sabul',
 'sadism',
 'sadist',
 'sarap',
 'scank',
 'scat',
 'schaffer',
 'scheiss',
 'schlampe',
 'schlong',
 'schmuck',
 'school',
 'screw',
 'screwing',
 'scrotum',
 'seks',
 'selangkang',
 'semen',
 'sempak',
 'senggama',
 'sepong',
 'setan',
 'setubuh',
 'sex',
 'sexy',
 'sharmuta',
 'sharmute',
 'shemale',
 'shipal',
 'shit',
 'shiter',
 'shits',
 'shitter',
 'shitty',
 'shity',
 'shitz',
 'shiz',
 'shyt',
 'shyte',
 'shytty',
 'shyty',
 'silet',
 'silit',
 'sinting',
 'sixty-nine',
 'sixtynine',
 'skanck',
 'skank',
 'skankee',
 'skankey',
 'skanks',
 'skanky',
 'skribz',
 'skurwysyn',
 'slag',
 'slut',
 'sluts',
 'slutty',
 'slutz',
 'smut',
 'sodomi',
 'sodomize',
 'sodomy',
 'softcore',
 'son-of-a-bitch',
 'spank',
 'spanked',
 'spanking',
 'sperm',
 'sphencter',
 'spic',
 'spierdalaj',
 'splooge',
 'squirt',
 'squirted',
 'squirting',
 'stfu',
 'strap-on',
 'strapon',
 'stres',
 'submissive',
 'suck',
 'suck-off',
 'sucked',
 'sucking',
 'sucks',
 'suicide',
 'suka',
 'taek',
 'tai',
 'tanpa busana',
 'taptei',
 'teets',
 'teez',
 'teho',
 'telanjang',
 'telaso',
 'temp',
 'testical',
 'testicle',
 'testicles',
 'tete',
 'tetek',
 'tewas',
 'three some',
 'threesome',
 'tit',
 'titit',
 'tits',
 'titt',
 'titties',
 'titty',
 'tittys',
 'togel',
 'toket',
 'tolol',
 'topless',
 'totong',
 'tranny',
 'transsexual',
 'transvestite',
 'tukar istri',
 'tukar pasangan',
 'turd',
 'tusbol',
 'twat',
 'twats',
 'twaty',
 'twink',
 'upskirt',
 'urin',
 'urinated',
 'urinating',
 'urination',
 'utek',
 'vagiina',
 'vagina',
 'vaginas',
 'vags',
 'vajina',
 'vibrator',
 'vittu',
 'vullva',
 'vulva',
 'wank',
 'wanking',
 'warez',
 'washu',
 'wasu',
 'wasuh',
 'watersports',
 'whoar',
 'whoor',
 'whore',
 'whores',
 'wichser',
 'woose',
 'wop',
 'wtf',
 'x-girl',
 'x-rated',
 'xrated',
 'xxx',
 'yateam',
 'yatim'}

# include_words = 


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
    text = stop_words_removal_question(text)
    return text

def delete_emojis_from_dict(text):
    for emoji in emoji_dict.keys():
        text = re.sub(re.escape(emoji), "", text)
    return text.strip()

def normalize_text(sentence):

    exclude_words = {'maaf', 'ganggu', 'pinggang', 'bangga', 'canggung', 'enggak', 'ngga', 'nggak', 'enggan', 'genggam', 'hingga', 'inggris', 'jingga', 'perunggu', 'ringgit', 'sehingga', 'tinggi', 'serangga', 'tangga', 'tanggap', 'tinggal', 'tanggal', 'anggun', 'app', 'booking', 'cheese', 'coffee', 'pizza', 'kiss', 'kesukaan', 'pelangaan'}
    # Split the sentence into words
    words = sentence.split()

    # Normalize each word in the sentence
    normalized_words = []
    for word in words:
        if word.lower() in exclude_words:
            normalized_words.append(word)
        else:
            # Restore the excluded patterns that were masked
            # Step 1: Normalize repeated characters except 'a' and 'g'
            normalized_word = re.sub(r'([^agAGzZfF])\1+', r'\1', word)

            # Step 2: Remove repeated 'a' or 'g' at the end, but keep single occurrences
            normalized_word = re.sub(r'([agAGzZ])\1+\Z', r'\1', normalized_word)
            normalized_words.append(normalized_word)

    # Join the normalized words back into a sentence
    return ' '.join(normalized_words)

def preprocess_text_delete_emoji_and_normalize(text):
    text = text.lower()
    text = delete_emojis_from_dict(text)
    text = re.sub(r'@\w+', '', text).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stop_words_removal_question(text)
    text = normalize_text(text)
    return text

def preprocess_text_and_normalize(text):
    text = text.lower()
    text = replace_emoji_with_word(text)
    text = re.sub(r'@\w+', '', text).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stop_words_removal_question(text)
    text = normalize_text(text)
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

from collections import defaultdict
from itertools import islice


def get_key_words_and_clean_up(texts, class_labels, stanza, tokenizer, model, preprocess=False):
    if preprocess:
        texts = [preprocess_text_delete_emoji_and_normalize(text) for text in texts]
    print(texts)
    nlp = stanza
    pos_dict = defaultdict(int)
    neg_dict = defaultdict(int)
    pos_nouns_adjs = defaultdict(int)
    neg_nouns_adjs = defaultdict(int)
    pos_negations = defaultdict(int)
    neg_negations = defaultdict(int)

    res_adjectives = set()
    res_nouns = set()
    
    for text, label in zip(texts, class_labels):
        previous_noun = None
        previous_negation = None  # Track negations
        doc = nlp(text)

        for sent in doc.sentences:
            for word in sent.words:
                word_text = word.text.lower()
                if len(word_text) == 1:
                    continue 

                word_text = re.sub(r'ny[ae]*$', '', word_text, flags=re.IGNORECASE)

                if word_text in exclude_words or word_text in exclude_nouns:
                    previous_noun = None
                    previous_negation = None
                    continue
                
                if word_text in negations:
                    # If negation is found, store it to pair with an adjective later
                    previous_negation = word_text

                elif word.upos == "ADJ" or word_text in adjectives:
                    if previous_negation:
                        # Negation-adjective phrase
                        neg_phrase = f"{previous_negation} {word_text}"
                        if label == 2:
                            pos_negations[neg_phrase] += 1
                        elif label == 0:
                            neg_negations[neg_phrase] += 1
                        previous_negation = None  # Reset negation after pairing
                        
                    elif previous_noun:
                        # Noun-adjective phrase
                        phrase = f"{previous_noun} {word_text}"
                        if label == 2:
                            pos_dict[phrase] += 1
                            pos_nouns_adjs[previous_noun] += 1
                            pos_nouns_adjs[word_text] += 1
                        elif label == 0:
                            neg_dict[phrase] += 1
                            neg_nouns_adjs[previous_noun] += 1
                            neg_nouns_adjs[word_text] += 1
                        previous_noun = None
                    else:
                        # Standalone adjective
                        res_adjectives.add(word_text)
                        if label == 2:
                            pos_dict[word_text] += 1
                            pos_nouns_adjs[word_text] += 1
                        elif label == 0:
                            neg_dict[word_text] += 1
                            neg_nouns_adjs[word_text] += 1


                elif word.upos == "NOUN" or word_text in nouns:
                    previous_noun = word_text
                    if label == 2:
                        pos_nouns_adjs[word_text] += 1
                    elif label == 0:
                        neg_nouns_adjs[word_text] += 1

    # Filter using your model predictions for better quality
    pos_arr, neg_arr = get_array_words(pos_dict, neg_dict)
    pos_dict = filter_with_model(pos_arr, pos_dict, tokenizer, model, label=2)
    neg_dict = filter_with_model(neg_arr, neg_dict, tokenizer, model, label=0)

    pos_nouns_adjs.update(pos_negations)
    neg_nouns_adjs.update(neg_negations)

    def custom_sort(item):
        key, count = item
        is_adj = key in res_adjectives
        return (-count, not is_adj)  # Prioritize high count, then adjectives

    # Update sorting logic for pos_nouns_adjs and neg_nouns_adjs
    pos_nouns_adjs = dict(sorted(pos_nouns_adjs.items(), key=lambda item: custom_sort(item)))
    neg_nouns_adjs = dict(sorted(neg_nouns_adjs.items(), key=lambda item: custom_sort(item)))

    pos_dict = dict(sorted(pos_dict.items(), key=lambda item: custom_sort(item)))
    neg_dict = dict(sorted(neg_dict.items(), key=lambda item: custom_sort(item)))

    if len(pos_nouns_adjs) > 15:
        pos_nouns_adjs = dict(islice(pos_nouns_adjs.items(), 15))  
    if len(neg_nouns_adjs) > 15:
        neg_nouns_adjs = dict(islice(neg_nouns_adjs.items(), 15))  

    if len(pos_dict) > 15:
        pos_dict = dict(islice(pos_dict.items(), 15))  
    if len(neg_dict) > 15:
        neg_dict = dict(islice(neg_dict.items(), 15))  
    return pos_dict, neg_dict, pos_nouns_adjs, neg_nouns_adjs

def filter_with_model(arr, word_dict, tokenizer, model, label):
    if len(arr) > 0:
        tokenized = tokenize_batch(arr, tokenizer)
        predictions = model.predict(tokenized)
        class_labels = np.argmax(predictions, axis=1)

        for i, pred_label in enumerate(class_labels):
            if pred_label != label:
                word = arr[i]
                del word_dict[word]
    return word_dict


# def process_key_words(pos_dict, neg_dict):
#     for (noun, adj), count in pos_dict:


def get_array_words(pos_common_words, neg_common_words):
    top_3_pos_words = [word for word, _ in pos_common_words.items()]
    top_3_neg_words = [word for word, _ in neg_common_words.items()]
    
    return top_3_pos_words, top_3_neg_words

def get_tag_words_bruh(texts, class_labels, stanza):
    pos_common_words, neg_common_words = analyze_sentiment(texts, class_labels, stanza)
    return get_array_words(pos_common_words, neg_common_words)

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

def get_questions_or_assistance(netraL_data, class_labels):
    questions_data = []
    for i, label in enumerate(class_labels):
        if label == 1:
            questions_data.append(netraL_data[i])

    return questions_data


def get_username(comments_to_idx, comments, usernames):
    extracted_username = []
    for comment in comments:
        pos_idx = comments_to_idx[comment]
        username = usernames[pos_idx]
        extracted_username.append(username)

    return extracted_username
    

def predict_assistance_batch(texts, model, tokenizer, preprocess=True, treshold=0.5):
    if preprocess:
        preprocessed_texts = [preprocess_text_question(text) for text in texts]
    else:
        preprocessed_texts = texts
    # print(preprocessed_texts)
    tokenized_texts = tokenize_batch(preprocessed_texts, tokenizer)
    predictions = model.predict(tokenized_texts)
    # print(predictions)
    question_labels = ["Bukan Minta Assistance", "Minta Assistance"]
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

import random

def limit_and_filter_comments_400(texts, class_labels):
    positive_comments = []
    negative_comments = []
    new_class_labels = []

    # Separate comments by class labels
    for i, label in enumerate(class_labels):
        if label == 2:  # Positive
            positive_comments.append(texts[i])
        elif label == 0:  # Negative
            negative_comments.append(texts[i])

    # If total comments are less than or equal to 400, return them as is
    if len(positive_comments) + len(negative_comments) <= 400:
        new_class_labels = [2] * len(positive_comments) + [0] * len(negative_comments)
        return positive_comments + negative_comments, new_class_labels

    # Calculate the reduction strategy
    if len(positive_comments) > len(negative_comments):
        keep_negative = min(len(negative_comments), 400)
        keep_positive = 400 - keep_negative
    elif len(negative_comments) > len(positive_comments):
        keep_positive = min(len(positive_comments), 400)
        keep_negative = 400 - keep_positive
    else:
        keep_positive = keep_negative = 200

    # Randomly sample comments to keep
    reduced_positive_comments = random.sample(positive_comments, min(keep_positive, len(positive_comments)))
    reduced_negative_comments = random.sample(negative_comments, min(keep_negative, len(negative_comments)))

    # Update class labels
    new_class_labels = [2] * len(reduced_positive_comments) + [0] * len(reduced_negative_comments)

    # Combine and shuffle the reduced comments and labels for fairness
    reduced_comments = reduced_positive_comments + reduced_negative_comments
    combined = list(zip(reduced_comments, new_class_labels))
    random.shuffle(combined)

    reduced_comments, new_class_labels = zip(*combined)
    return list(reduced_comments), list(new_class_labels)


""" VERTEX AI """


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud_credentials\c242-ps228-capstone-team-b04f8a78f4f7.json"

def load_vertex_model():     
    vertexai.init(project="132823030367", location="us-central1")
   
    model = GenerativeModel(
        "projects/132823030367/locations/us-central1/endpoints/5514526901831467008",
    )

    return model

def replace_bad_words(texts_array):
    
    def clean_sentence(sentence):
        return ' '.join([word if word not in bad_words else 'Buruk' for word in sentence.split()])

    return [clean_sentence(sentence) for sentence in texts_array]

def create_gen_ai_input(texts_array):
    texts_array = replace_bad_words(texts_array)
    reduced_comments = []
    for i, comment in enumerate(texts_array):
        truncated_comment = comment
        if len(comment) > 150:
            truncated_comment = comment[:150]
        reduced_comments.append(truncated_comment)

    formatted_commment = ""
    for i, comment in enumerate(reduced_comments):
        if i != len(reduced_comments) - 1:
            formatted_commment += comment + '\n' 
        else:
            formatted_commment += comment

    if len(formatted_commment) > 25000:
        formatted_commment = formatted_commment[:25000]

    return formatted_commment

def generate_resume(prompt, model):
    chat = model.start_chat()
    generation_config = {
        "max_output_tokens": 2048,
        "temperature": 0.8,
        "top_p": 1,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE
        ),
    ]

    response = chat.send_message(prompt, generation_config=generation_config, safety_settings=safety_settings)    
    print("Response: ", response.text)

    return response.text

def decode_emoji(dict):
    for words, count in dict.items():
        words_arr = words.split()
        for i, word in enumerate(words_arr):
            if word in text_to_emoji_dict:
                emoji = text_to_emoji_dict[word]
                words_arr[i] = emoji
        new_words = ' '.join(words_arr)
        dict[new_words] = dict.pop(words)

    return dict

    # new_texts = []
    # for i, text in enumerate(texts):
    #     words = text.split()
    #     new_text = []
    #     for i, word in enumerate(words):
    #         if word in text_to_emoji_dict:
    #             emoji = text_to_emoji_dict[word]
    #             new_text.append(emoji)
    #         else:
    #             new_text.append(word)

    #     new_texts.append(new_text)
    # return new_texts