import os
import time
import json
import sys
import subprocess

# ==============================================================================
# SEKCJA: Automatyczne sprawdzanie i instalowanie brakujących bibliotek
# ==============================================================================
def check_and_install_dependencies():
    required_packages = {
        'torch': 'torch', 'datasets': 'datasets', 'transformers': 'transformers',
        'huggingface-hub': 'huggingface_hub', 'accelerate': 'accelerate'
    }
    print("--- Sprawdzanie wymaganych bibliotek ---")
    all_installed = True
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            all_installed = False
            print(f"⚠️  Brak biblioteki '{package_name}'. Rozpoczynam instalację...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✅  Pomyślnie zainstalowano '{package_name}'.")
            except subprocess.CalledProcessError as e:
                print(f"❌  BŁĄD: Nie udało się zainstalować '{package_name}'. Proszę zainstalować ją ręcznie.")
                sys.exit(1)
    if all_installed:
        print("✅  Wszystkie wymagane biblioteki są już zainstalowane.")
    print("----------------------------------------\n")

check_and_install_dependencies()

# ==============================================================================
# GŁÓWNE IMPORTY
# ==============================================================================
import torch
from datasets import load_dataset
from transformers import (
    GemmaConfig, GemmaForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback, TrainerState, TrainerControl
)
import huggingface_hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
HF_TOKEN = "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"
BLOCK_SIZE = 512
CONFIG_FILE = ".last_model_path.txt"
LATEST_TRAINED_MODEL_PATH = None

def save_last_model_path(path):
    with open(CONFIG_FILE, 'w') as f: f.write(path)

def load_last_model_path():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            path = f.read().strip()
            if os.path.isdir(path): return path
    return None

class Colors:
    BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'

class ProgressCallback(TrainerCallback):
    def __init__(self): self.step_start_time = 0
    def on_step_begin(self, args, state, control, **kwargs): self.step_start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        duration = time.time() - self.step_start_time
        tokens_in_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * BLOCK_SIZE
        tokens_per_second = tokens_in_step / duration
        print(f"  [Krok {state.global_step}] Czas: {duration:.2f}s | Prędkość: {int(tokens_per_second):,} t/s".replace(",", " "), end="\r")

def run_pretraining():
    global LATEST_TRAINED_MODEL_PATH
    MODEL_CONFIG_PATH = "./moj_model"
    DATASET_NAME = "chrisociepa/wikipedia-pl-20230401"
    DATASET_PERCENTAGE = 100
    NUM_PROC_MAP = 8
    NUM_LAYERS = 4
    OUTPUT_DIR = f"./wytrenowany_model_PL_{NUM_LAYERS}L"
    
    # --- Architektura i Model ---
    with open(os.path.join(MODEL_CONFIG_PATH, "config.json"), 'r') as f:
        config = GemmaConfig(**json.load(f)['text_config'])
    config.num_hidden_layers = NUM_LAYERS
    model = GemmaForCausalLM(config)
    print(f"{Colors.YELLOW}   -> Model z {NUM_LAYERS} warstwami. Liczba parametrów: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} mld{Colors.ENDC}")
    
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # --- Dane ---
    dataset = load_dataset(DATASET_NAME, split="train").select(range(int(len(load_dataset(DATASET_NAME, split="train")) * (DATASET_PERCENTAGE / 100))))
    def tokenize_function(e): return tokenizer(e["text"], truncation=True, max_length=BLOCK_SIZE)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=NUM_PROC_MAP, remove_columns=dataset.column_names)
    def group_texts(e):
        concatenated = {k: sum(e[k], []) for k in e.keys()}
        total_length = len(concatenated[list(e.keys())[0]])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=NUM_PROC_MAP)

    # --- Trening ---
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, overwrite_output_dir=True, num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=16, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), gradient_checkpointing=True, logging_steps=20, save_steps=1000, save_total_limit=2)
    trainer = Trainer(model=model, args=training_args, train_dataset=lm_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), callbacks=[ProgressCallback()])
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}!!! ROZPOCZYNAM PRE-TRENING !!!{Colors.ENDC}")
    trainer.train()
    print(f"\n{Colors.BOLD}{Colors.GREEN}!!! PRE-TRENING ZAKOŃCZONY !!!{Colors.ENDC}\n")
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    LATEST_TRAINED_MODEL_PATH = OUTPUT_DIR
    save_last_model_path(OUTPUT_DIR)
    print(f"{Colors.YELLOW}Model zapisany i zapamiętany w: {OUTPUT_DIR}{Colors.ENDC}")


# ### NAPRAWIONA FUNKCJA WYSYŁANIA ###
def upload_to_hub():
    """Automatycznie znajduje ostatni model i wysyła go na Hugging Face Hub."""
    print("\n" + "="*50)
    print(f"{Colors.BOLD}{Colors.BLUE}--- WYSYŁANIE MODELU NA HUGGING FACE HUB ---{Colors.ENDC}")
    print("="*50 + "\n")

    if LATEST_TRAINED_MODEL_PATH is None:
        print(f"{Colors.RED}Błąd: Nie znaleziono ścieżki do wytrenowanego modelu.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Uruchom najpierw trening (opcja 1).{Colors.ENDC}")
        return

    model_path = LATEST_TRAINED_MODEL_PATH
    print(f"{Colors.BLUE}Używam modelu z ostatnio zapamiętanej ścieżki: {model_path}{Colors.ENDC}")

    repo_name = input(f"{Colors.YELLOW}Podaj nazwę repozytorium na Hugging Face (np. TwojaNazwa/NazwaModelu): {Colors.ENDC}")
    if not repo_name or "/" not in repo_name:
        print(f"{Colors.RED}Błąd: Nieprawidłowa nazwa repozytorium. Musi być w formacie 'uzytkownik/nazwa'.{Colors.ENDC}")
        return

    try:
        print(f"\n{Colors.BLUE}Logowanie i wysyłanie plików do repozytorium '{repo_name}'...{Colors.ENDC}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GemmaForCausalLM.from_pretrained(model_path)
        
        # Użycie tokena bezpośrednio w metodach push_to_hub
        tokenizer.push_to_hub(repo_id=repo_name, token=HF_TOKEN)
        model.push_to_hub(repo_id=repo_name, token=HF_TOKEN)
        
        print("\n" + "="*50)
        print(f"{Colors.BOLD}{Colors.GREEN}!!! MODEL POMYŚLNIE WYSŁANY !!!{Colors.ENDC}")
        print(f"{Colors.YELLOW}Możesz go znaleźć pod adresem: https://huggingface.co/{repo_name}{Colors.ENDC}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"{Colors.RED}Wystąpił nieoczekiwany błąd: {e}{Colors.ENDC}")

def test_model():
    """Ładuje model ze stałej ścieżki i pozwala na testowanie."""
    MODEL_PATH = "./wytrenowany_model_PL_4L"
    print("\n" + "="*50)
    print(f"{Colors.BOLD}{Colors.BLUE}--- TESTOWANIE MODELU ---{Colors.ENDC}")
    print(f"{Colors.YELLOW}Próba załadowania modelu z: {MODEL_PATH}{Colors.ENDC}")
    print("="*50 + "\n")

    if not os.path.isdir(MODEL_PATH):
        print(f"{Colors.RED}BŁĄD: Nie znaleziono folderu z modelem '{MODEL_PATH}'.{Colors.ENDC}")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = GemmaForCausalLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print(f"{Colors.GREEN}Model załadowany pomyślnie na: {device}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Błąd podczas ładowania modelu: {e}{Colors.ENDC}")
        return

    while True:
        prompt = input(f"\n{Colors.YELLOW}Wpisz początek zdania (lub 'wyjdz'): {Colors.ENDC}")
        if prompt.lower() in ['wyjdz', 'exit', 'quit']: break
        max_new_tokens = 150
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "-"*50 + f"\n{Colors.BOLD}{Colors.GREEN}--- WYGENEROWANY TEKST ---{Colors.ENDC}\n{generated_text}\n" + "-"*50)

if __name__ == "__main__":
    LATEST_TRAINED_MODEL_PATH = load_last_model_path()
    while True:
        print("\n" + "="*50)
        print(f"{Colors.BOLD}{Colors.BLUE}--- MENU GŁÓWNE ---{Colors.ENDC}")
        if LATEST_TRAINED_MODEL_PATH: print(f"[{Colors.GREEN}INFO{Colors.ENDC}] Zapamiętany model: {LATEST_TRAINED_MODEL_PATH}")
        print("1. Trenuj nowy model")
        print("2. Wyślij ostatni model na Hugging Face")
        print("3. Przetestuj model z folderu ./wytrenowany_model_PL_4L")
        print("4. Zakończ program")
        print("="*50)
        choice = input(f"{Colors.YELLOW}Wybierz opcję (1-4): {Colors.ENDC}")
        if choice == '1': run_pretraining()
        elif choice == '2': upload_to_hub()
        elif choice == '3': test_model()
        elif choice == '4': break
        else: print(f"{Colors.RED}Nieprawidłowy wybór.{Colors.ENDC}")
    print(f"{Colors.GREEN}Do widzenia!{Colors.ENDC}")