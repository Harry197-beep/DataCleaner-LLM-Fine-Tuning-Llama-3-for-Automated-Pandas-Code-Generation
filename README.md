
🧹 DataCleaner-LLM: Fine-Tuning Llama-3 for Automated Pandas Code Generation

PythonLlama 3UnslothQLoRACompute

🎯 Overview

Data analysts spend up to 80% of their time cleaning messy data. While generic LLMs (like ChatGPT) can write Pandas code, they often hallucinate methods, ignore edge cases, or write unsafe, non-idempotent code (e.g., overusing inplace=True).

DataCleaner-LLM is a fine-tuned version of Llama-3-8B-Instruct designed to act as a strict "Pandas Purist." Given a text description of a messy dataset schema, it generates production-ready, method-chained, and highly robust Python/Pandas cleaning scripts.

🤔 Why Fine-Tuning Instead of Prompting?

Prompt engineering is great for general tasks, but terrible for enforcing strict coding standards. By fine-tuning this model on 1,000 high-quality examples, I forced the model to:

Never use inplace=True (which causes chained assignment issues).
Always use method chaining for cleaner, more readable pipelines.
Use safe coercions (e.g., pd.to_datetime(..., errors='coerce')) instead of failing on bad data.
⚙️ Technical Architecture & Methodology

This project was built with a strict $0 budget constraint, proving that state-of-the-art fine-tuning is accessible to anyone.

Base Model: unsloth/llama-3-8b-Instruct
Fine-Tuning Technique: QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization. This drastically reduces memory usage while retaining base model capabilities.
Compute: Trained entirely on a free Google Colab T4 GPU (16GB VRAM) using the Unsloth library for 2x faster training.
Dataset Generation: To avoid API costs, I wrote a custom Python script to programmatically generate 1,000 synthetic training pairs [Messy Schema Description] -> [Perfect Pandas Code].
💻 Example Input / Output

Input Prompt:

"I have a dataframe. The 'Purchase_Date' column has mixed formats like 'MM/DD/YYYY' and 'Jan 1st, 2023'. The 'Price' column has dollar signs and commas (e.g., '$1,200.50'). There are about 10% missing values in the 'User_ID' column."

Model Output:

import pandas as pddef clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:    return (        df        # Safe date coercion, turning unparsable dates into NaT        .assign(Purchase_Date=lambda x: pd.to_datetime(x['Purchase_Date'], errors='coerce'))        # Remove string characters and convert to float        .assign(Price=lambda x: x['Price'].str.replace(r'[$,]', '', regex=True).astype(float))        # Fill missing User_IDs with a placeholder to maintain data type        .assign(User_ID=lambda x: x['User_ID'].fillna('UNKNOWN'))    )
🚀 How to Run

You can run this project in under 15 minutes for free.

Open the Notebook: Click Open In Colab (Replace this link with your actual Colab notebook link)
Change Runtime: Go to Runtime -> Change runtime type -> T4 GPU.
Run Cells: Execute the cells sequentially. It will:
 Install Unsloth.
 Generate the synthetic dataset.
 Fine-tune the model (takes ~10 mins on T4).
 Launch a local Gradio interface to test it yourself.
📂 Repository Structure

text

├── README.md               # You are here
├── DataCleaner_Finetune.ipynb # The main Colab notebook
├── requirements.txt        # Core dependencies (unsloth, trl, peft, etc.)
└── assets/                 # Images/GIFs for the README
🛠️ Tech Stack

 Language: Python
 ML Frameworks: PyTorch, Hugging Face transformers, trl, peft
 Optimization: Unsloth, BitsAndBytes (4-bit quantization)
 UI: Gradio
📈 Future Improvements

 Expand training data to include PySpark cleaning logic for big data environments.
 Add support for multi-table joining logic based on schema descriptions.
 Deploy the fine-tuned LoRA weights as a lightweight API endpoint using FastAPI.
📜 License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: Base Llama-3 model is governed by Meta's Llama 3 Community License).
