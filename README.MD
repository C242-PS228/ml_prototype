# Project Setup Instructions

## Table of Contents
- [Project Setup Instructions](#project-setup-instructions)
  - [Table of Contents](#table-of-contents)
  - [Setup Instructions](#setup-instructions)
  - [Important Links](#important-links)
    - [Notebooks](#notebooks)
    - [Resources](#resources)
  - [Model Performance](#model-performance)
  - [Generative Model](#generative-model)

---

## Setup Instructions

1. **Clone the Repository**:  
   Clone the repository to your local machine:  
   `git clone <repository-url>`

2. **Create a Folder for Models**:  
   Create a folder named `model` in the root directory.

3. **Download Model Files**:  
   Download the following files from the [Final Models Drive link](https://drive.google.com/drive/folders/1DtEKk2ln1VGZ5VwUEM_4LWBwpFi5PjDo?usp=drive_link):  
   - `bert_attention_v12.h5`  
   - `question_bert.h5`  
   - `assistance_bert.h5`  
   Move these files to the `model` folder.

4. **Set Up a Virtual Environment**:  
   Create and activate a virtual environment:  
   `virtualenv venv`  
   `source venv/bin/activate` (On Windows: `venv\Scripts\activate`)

5. **Install Dependencies**:  
   Install the required Python libraries:  
   `pip install -r requirements.txt`  
   `pip install stanza`

6. **Download Stanza Models**:  
   Download Stanza models from [this Drive link](https://drive.google.com/drive/folders/1zvIxOpHYeQBYG9hzXQgWGMCGIbuQjK9X?usp=sharing).  
   Place the downloaded files in the following folder structure:  
   `root -> stanza_models`

7. **Run the API**:  
   Start the API server:  
   `uvicorn api_ml:app --reload`

8. **Run the Web Prototype (Optional)**:  
   If you want:  
   `python main.py`


---

## Important Links

### Notebooks
- **Sentiment Analysis Training**: [Open in Colab](https://colab.research.google.com/drive/1aoz43nCdwJqJQ2qdRWsQZYt4grmwuUdf)  
- **Question Detection Training**: [Open in Colab](https://colab.research.google.com/drive/1VVAZ2F7JWSD5D7doMzM4RlGbnvnXbpvc)  
- **Asking for Assistance Training**: [Open in Colab](https://colab.research.google.com/drive/1KmePZIYqTXtmCb2HQsHn-wbbotMZNmVT)  
- **Data Preprocessing**: [Open in Colab](https://colab.research.google.com/drive/1C0g-30zh4MayyeNWv0HwQtWILgKRGZXs)
- **Gen AI Vertex**: [Open in Colab](https://colab.research.google.com/drive/109IyEBePI-CU8eMx5uMmCpW1mXh0hdih?usp=sharing)

### Resources
- **Final Models (.h5)**: [Download from Drive](https://drive.google.com/drive/folders/1DtEKk2ln1VGZ5VwUEM_4LWBwpFi5PjDo?usp=drive_link)  
- **Dataset**: [Download from Drive](https://drive.google.com/drive/folders/13yKcgvN8_X9MSDDmGxhxxrdw8j8y_FfA?usp=drive_link)

---

## Model Performance

| **Task**                 | **Accuracy** |
|--------------------------|--------------|
| Sentiment Analysis       | 88%          |
| Question Detection       | 84%          |
| Ask Assistance Detection | 87%          |

---

## Generative Model

| **Task** | **Base Model**      |
|----------|---------------------|
| Resume   | Gemini-1.0-pro-002  |