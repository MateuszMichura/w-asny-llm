import os
import time
import json
import sys
import subprocess
import math
import glob
import shutil

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
# GŁÓWNE IMPORTY I LOGOWANIE
# ==============================================================================
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    GemmaConfig, GemmaForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from huggingface_hub import repo_info, login

# ---> PAMIĘTAJ O ZMIANIE TEGO TOKENA NA NOWY I BEZPIECZNY <---
TWOJ_TOKEN_HUGGING_FACE = "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" 

try:
    login(token=TWOJ_TOKEN_HUGGING_FACE)
    print(f"\033[92m✅ Pomyślnie zalogowano na konto Hugging Face.\033[0m")
except Exception as e:
    print(f"\033[91m❌ BŁĄD logowania do Hugging Face: {e}\033[0m")
    print(f"\033[93mUpewnij się, że token w zmiennej TWOJ_TOKEN_HUGGING_FACE jest poprawny.\033[0m")
    sys.exit(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# ZMIENNE GLOBALNE I FUNKCJE POMOCNICZE
# ==============================================================================
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

def get_dir_size_gb(path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

class ProgressCallback(TrainerCallback):
    def __init__(self): self.step_start_time = 0
    def on_step_begin(self, args, state, control, **kwargs): self.step_start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        duration = time.time() - self.step_start_time
        tokens_in_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * BLOCK_SIZE
        tokens_per_second = tokens_in_step / duration
        print(f"  [Krok {state.global_step}] Czas: {duration:.2f}s | Prędkość: {int(tokens_per_second):,} t/s".replace(",", " "), end="\r")

def get_dataset_info(repo_id, lang_prefix="pl/"):
    try:
        info = repo_info(repo_id, repo_type="dataset", files_metadata=True)
        parquet_files = [f for f in info.siblings if f.rfilename.startswith(lang_prefix) and f.rfilename.endswith('.parquet')]
        if not parquet_files: return 0, 0, 0
        total_files = len(parquet_files)
        total_size_bytes = sum(f.size for f in parquet_files if f.size is not None)
        total_size_gb = total_size_bytes / (1024**3)
        avg_file_size_gb = total_size_gb / total_files if total_files > 0 else 0
        return total_files, total_size_gb, avg_file_size_gb
    except Exception as e:
        print(f"{Colors.RED}BŁĄD: Nie udało się pobrać informacji o zbiorze danych: {e}{Colors.ENDC}")
        return None, None, None

# ==============================================================================
# GŁÓWNE FUNKCJE APLIKACJI
# ==============================================================================
def run_pretraining():
    global LATEST_TRAINED_MODEL_PATH
    
    # ZMIANA: Usunąłem "./" z nazwy folderu. Jest czyściej i bezpieczniej.
    MODEL_CONFIG_PATH = "moj_model"
    BASE_MODEL_ID = "google/gemma-2b" # Model, z którego pobierzemy startową konfigurację
    DATASET_NAME = "uonlp/CulturaX"
    NUM_LAYERS = 4
    OUTPUT_DIR = f"./wytrenowany_model_PL_{NUM_LAYERS}L"
    PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_dataset")

    # ==========================================================================
    # ZMIANA: NOWA SEKCJA - AUTOMATYCZNE PRZYGOTOWANIE MODELU BAZOWEGO
    # ==========================================================================
    # Sprawdzamy, czy folder z konfiguracją istnieje i zawiera potrzebne pliki.
    if not os.path.isdir(MODEL_CONFIG_PATH) or not os.path.exists(os.path.join(MODEL_CONFIG_PATH, "tokenizer.json")):
        print(f"{Colors.YELLOW}INFO: Folder '{MODEL_CONFIG_PATH}' jest pusty lub nie istnieje.{Colors.ENDC}")
        print(f"{Colors.BLUE}Pobieram pliki bazowe dla modelu '{BASE_MODEL_ID}' z Hugging Face...{Colors.ENDC}")
        try:
            os.makedirs(MODEL_CONFIG_PATH, exist_ok=True)
            # Pobieramy i zapisujemy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            tokenizer.save_pretrained(MODEL_CONFIG_PATH)
            # Pobieramy i zapisujemy konfigurację w formacie wymaganym przez skrypt
            config = GemmaConfig.from_pretrained(BASE_MODEL_ID)
            config_dict_for_script = {"text_config": config.to_dict()}
            with open(os.path.join(MODEL_CONFIG_PATH, "config.json"), 'w') as f:
                json.dump(config_dict_for_script, f, indent=4)
            print(f"{Colors.GREEN}✅ Pomyślnie pobrano i zapisano pliki startowe w folderze '{MODEL_CONFIG_PATH}'.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ KRYTYCZNY BŁĄD: Nie udało się pobrać plików bazowych. Sprawdź połączenie z internetem.{Colors.ENDC}")
            print(f"Szczegóły błędu: {e}")
            return # Zatrzymujemy działanie funkcji
    else:
        print(f"{Colors.GREEN}INFO: Znaleziono lokalną konfigurację modelu w '{MODEL_CONFIG_PATH}'.{Colors.ENDC}")
    # ==========================================================================
    # KONIEC NOWEJ SEKCJI
    # ==========================================================================

    resume_from_checkpoint = False
    
    if os.path.isdir(OUTPUT_DIR) and glob.glob(os.path.join(OUTPUT_DIR, f"{PREFIX_CHECKPOINT_DIR}-*")):
        print(f"{Colors.YELLOW}INFO: Znaleziono niedokończony trening w '{OUTPUT_DIR}'.{Colors.ENDC}")
        while True:
            choice = input(f"{Colors.YELLOW}Wybierz akcję: [1] Wznów, [2] Zacznij od nowa (usunie postęp), [3] Anuluj: {Colors.ENDC}")
            if choice == '1':
                resume_from_checkpoint = True
                print(f"{Colors.GREEN}Wybrano wznawianie treningu...{Colors.ENDC}")
                break
            elif choice == '2':
                print(f"{Colors.RED}Wybrano rozpoczęcie od nowa. Usuwanie starego postępu...{Colors.ENDC}")
                shutil.rmtree(OUTPUT_DIR)
                break
            elif choice == '3':
                print("Anulowano.")
                return
            else:
                print(f"{Colors.RED}Nieprawidłowy wybór.{Colors.ENDC}")

    if not resume_from_checkpoint:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"{Colors.BLUE}Sprawdzanie informacji o zbiorze danych '{DATASET_NAME}'...{Colors.ENDC}")
        total_files, total_size_gb, avg_file_size_gb = get_dataset_info(DATASET_NAME)
        
        if total_files is None or total_files == 0: return
        print(f"{Colors.GREEN}INFO: Cały polski zbiór danych składa się z {total_files} plików ({total_size_gb:.2f} GB).")
        print(f"      -> Każdy plik ma średnio {avg_file_size_gb:.2f} GB.{Colors.ENDC}")

        num_files_to_download = 0
        while True:
            try:
                choice_str = input(f"\n{Colors.YELLOW}Podaj, ile plików chcesz pobrać (1-{total_files}). Wpisz 'anuluj', aby wrócić: {Colors.ENDC}")
                if choice_str.lower() == 'anuluj': return
                choice_files = int(choice_str)
                if 1 <= choice_files <= total_files:
                    num_files_to_download = choice_files
                    break
                else: print(f"{Colors.RED}Błąd: Wprowadź liczbę od 1 do {total_files}.{Colors.ENDC}")
            except ValueError: print(f"{Colors.RED}Błąd: To nie jest poprawna liczba.{Colors.ENDC}")

        data_files = [f"pl/pl_part_{i:05d}.parquet" for i in range(num_files_to_download)]
        print(f"{Colors.BLUE}INFO: Rozpoczynam pobieranie i przetwarzanie danych...{Colors.ENDC}")
        try:
            # Teraz ta linia zadziała, bo folder MODEL_CONFIG_PATH jest już gotowy
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_PATH)
            
            dataset = load_dataset(DATASET_NAME, data_files=data_files, split="train", streaming=False)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            def tokenize_function(e): return tokenizer(e["text"], truncation=True, max_length=BLOCK_SIZE)
            tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=dataset.column_names)
            def group_texts(e):
                concatenated = {k: sum(e[k], []) for k in e.keys()}
                total_length = len(concatenated[list(e.keys())[0]])
                total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
                result = {k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)] for k, t in concatenated.items()}
                result["labels"] = result["input_ids"].copy()
                return result
            lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=os.cpu_count())
            print(f"{Colors.BLUE}Zapisywanie przetworzonego zbioru danych na dysk...{Colors.ENDC}")
            lm_dataset.save_to_disk(PROCESSED_DATA_DIR)
            data_size = get_dir_size_gb(PROCESSED_DATA_DIR)
            print(f"{Colors.GREEN}✅ Zapisano. Przetworzone dane zajmują {data_size:.2f} GB miejsca na dysku.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ BŁĄD podczas przetwarzania danych: {e}{Colors.ENDC}")
            return
            
    with open(os.path.join(MODEL_CONFIG_PATH, "config.json"), 'r') as f:
        config = GemmaConfig(**json.load(f)['text_config'])
    config.num_hidden_layers = NUM_LAYERS
    model = GemmaForCausalLM(config)
    print(f"{Colors.YELLOW}   -> Model z {NUM_LAYERS} warstwami. Liczba parametrów: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} mld{Colors.ENDC}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"{Colors.BLUE}INFO: Ładowanie przetworzonego zbioru danych z dysku...{Colors.ENDC}")
    if not os.path.isdir(PROCESSED_DATA_DIR):
        print(f"{Colors.RED}BŁĄD: Nie znaleziono folderu z przetworzonymi danymi: {PROCESSED_DATA_DIR}{Colors.ENDC}")
        return
    lm_dataset = load_from_disk(PROCESSED_DATA_DIR)
    data_size = get_dir_size_gb(PROCESSED_DATA_DIR)
    print(f"{Colors.GREEN}✅ Załadowano dane ({data_size:.2f} GB).{Colors.ENDC}")
    
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, overwrite_output_dir=False, num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=16, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), gradient_checkpointing=True, logging_steps=20, save_steps=1000, save_total_limit=2)
    trainer = Trainer(model=model, args=training_args, train_dataset=lm_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), callbacks=[ProgressCallback()])

    try:
        print(f"\n{Colors.BOLD}{Colors.GREEN}!!! ROZPOCZYNAM PRE-TRENING !!!{Colors.ENDC}")
        print(f"{Colors.YELLOW}Naciśnij Ctrl+C w dowolnym momencie, aby bezpiecznie przerwać i zapisać model.{Colors.ENDC}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(f"\n{Colors.BOLD}{Colors.GREEN}!!! PRE-TRENING ZAKOŃCZONY !!!{Colors.ENDC}\n")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        LATEST_TRAINED_MODEL_PATH = OUTPUT_DIR
        save_last_model_path(OUTPUT_DIR)
        print(f"{Colors.YELLOW}Model końcowy zapisany w: {OUTPUT_DIR}{Colors.ENDC}")
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Przerwano trening przez użytkownika (Ctrl+C).{Colors.ENDC}")
        print(f"{Colors.YELLOW}Zapisywanie pełnego stanu trenera...{Colors.ENDC}")
        trainer.save_state()
        trainer.save_model() 
        checkpoint_folder = os.path.join(OUTPUT_DIR, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}")
        if not os.path.exists(checkpoint_folder):
             os.makedirs(checkpoint_folder, exist_ok=True)
             shutil.copy(os.path.join(OUTPUT_DIR, "trainer_state.json"), checkpoint_folder)
             shutil.copy(os.path.join(OUTPUT_DIR, "optimizer.pt"), checkpoint_folder)
             shutil.copy(os.path.join(OUTPUT_DIR, "scheduler.pt"), checkpoint_folder)

        checkpoint_size = get_dir_size_gb(OUTPUT_DIR)
        print(f"{Colors.GREEN}✅ Stan treningu i model zapisany ({checkpoint_size:.2f} GB). Program zostanie zamknięty.{Colors.ENDC}")
        save_last_model_path(OUTPUT_DIR)
        sys.exit(0)

def upload_to_hub():
    print("\n" + "="*50)
    print(f"{Colors.BOLD}{Colors.BLUE}--- WYSYŁANIE MODELU NA HUGGING FACE HUB ---{Colors.ENDC}")
    print("="*50 + "\n")
    model_path_to_upload = load_last_model_path()
    if model_path_to_upload is None:
        print(f"{Colors.RED}Błąd: Nie znaleziono ścieżki do ostatnio trenowanego modelu.{Colors.ENDC}")
        return
    print(f"{Colors.BLUE}Używam modelu z ostatnio zapamiętanej ścieżki: {model_path_to_upload}{Colors.ENDC}")
    repo_name = input(f"{Colors.YELLOW}Podaj nazwę repozytorium na Hugging Face (np. TwojaNazwa/NazwaModelu): {Colors.ENDC}")
    if not repo_name or "/" not in repo_name:
        print(f"{Colors.RED}Błąd: Nieprawidłowa nazwa repozytorium.{Colors.ENDC}")
        return
    try:
        print(f"\n{Colors.BLUE}Wysyłanie plików do repozytorium '{repo_name}'...{Colors.ENDC}")
        tokenizer = AutoTokenizer.from_pretrained(model_path_to_upload)
        model = GemmaForCausalLM.from_pretrained(model_path_to_upload)
        tokenizer.push_to_hub(repo_id=repo_name)
        model.push_to_hub(repo_id=repo_name)
        print(f"\n{Colors.BOLD}{Colors.GREEN}!!! MODEL POMYŚLNIE WYSŁANY !!!{Colors.ENDC}")
        print(f"{Colors.YELLOW}Możesz go znaleźć pod adresem: https://huggingface.co/{repo_name}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Wystąpił nieoczekiwany błąd: {e}{Colors.ENDC}")

def test_model():
    # ZMIANA: Sprawdzamy najpierw ostatnio trenowany model. Jeśli go nie ma, używamy domyślnego.
    model_path_to_test = load_last_model_path()
    if model_path_to_test is None:
        model_path_to_test = "./wytrenowany_model_PL_4L" # Domyślna ścieżka, jeśli nic nie trenowano
    
    print("\n" + "="*50)
    print(f"{Colors.BOLD}{Colors.BLUE}--- TESTOWANIE MODELU ---{Colors.ENDC}")
    print(f"{Colors.YELLOW}Próba załadowania modelu z: {model_path_to_test}{Colors.ENDC}")
    print("="*50 + "\n")
    
    if not os.path.isdir(model_path_to_test):
        print(f"{Colors.RED}BŁĄD: Nie znaleziono folderu z modelem '{model_path_to_test}'.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Upewnij się, że model został już wytrenowany lub zmień ścieżkę w kodzie.{Colors.ENDC}")
        return
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path_to_test)
        model = GemmaForCausalLM.from_pretrained(model_path_to_test).to(device)
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

# ==============================================================================
# GŁÓWNA PĘTLA PROGRAMU
# ==============================================================================
if __name__ == "__main__":
    while True:
        LATEST_TRAINED_MODEL_PATH = load_last_model_path()
        print("\n" + "="*50)
        print(f"{Colors.BOLD}{Colors.BLUE}--- MENU GŁÓWNE ---{Colors.ENDC}")
        if LATEST_TRAINED_MODEL_PATH: print(f"[{Colors.GREEN}INFO{Colors.ENDC}] Zapamiętany model: {LATEST_TRAINED_MODEL_PATH}")
        print("1. Trenuj lub wznów trening modelu")
        print("2. Wyślij ostatni model na Hugging Face")
        print("3. Przetestuj ostatnio trenowany model") # ZMIANA: Lepszy opis
        print("4. Zakończ program")
        print("="*50)
        choice = input(f"{Colors.YELLOW}Wybierz opcję (1-4): {Colors.ENDC}")
        if choice == '1': run_pretraining()
        elif choice == '2': upload_to_hub()
        elif choice == '3': test_model()
        elif choice == '4': break
        else: print(f"{Colors.RED}Nieprawidłowy wybór.{Colors.ENDC}")
    print(f"{Colors.GREEN}Do widzenia!{Colors.ENDC}")
