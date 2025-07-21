## Language Translator Using Google API'S. 

import tkinter as tk
from tkinter import ttk, messagebox, filedialog           # GUI toolkit
from deep_translator import GoogleTranslator
import pyttsx3                                         # convert text to Speech
import threading                                       # Thread Based Parallalism
import hashlib
import os                                            # we are import os it can be use to stre the our translated file 
import json
from concurrent.futures import ThreadPoolExecutor

# **************************************** Voice Engine Setup ************************************ #
engine = pyttsx3.init()

def set_female_voice():
    voices = engine.getProperty('voices')
    for voice in voices:
        if "zira" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            return
    if voices:
        engine.setProperty('voice', voices[0].id)

set_female_voice()

#************************************** Language Cache ******************************************* #
def load_language_list():
    cache_file = "languages_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        translator = GoogleTranslator(source='auto', target='en')
        languages = translator.get_supported_languages(as_dict=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(languages, f)
        return languages

languages = load_language_list()
language_codes = list(languages.keys())

# *************************************** Globals *********************************************** #
translation_cache = {}
history = []
executor = ThreadPoolExecutor(max_workers=3)
is_translating = False

# ************************************** GUI Setup *********************************************/
root = tk.Tk()
root.geometry("600x650")
root.title("Language Translator")
root.configure(bg="#caffe2")
root.resizable(0, 0)

def update_status(msg):
    root.after(0, lambda: status_var.set(msg))

def make_cache_key(source, target, text):
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return (source, target, text_hash)

 #****************************** ******* Actions *************************************************/
def translate_text():
    global is_translating
    if is_translating:
        return
    is_translating = True

    def do_translate():
        global is_translating
        source = source_lang.get()
        target = target_lang.get()
        text = input_text.get("1.0", tk.END).strip()

        if not text:
            root.after(0, lambda: messagebox.showwarning("Input Error", "Please enter text to translate."))
            update_status("Ready")
            is_translating = False
            return

        update_status("Translating...")
        if auto_detect_var.get():
            source = 'auto'

        key = make_cache_key(source, target, text)
        if key in translation_cache:
            translated = translation_cache[key]
        else:
            try:
                translated = GoogleTranslator(source=source, target=target).translate(text)
                translation_cache[key] = translated
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Translation Error", str(e)))
                update_status("Ready")
                is_translating = False
                return

        history.append((text, translated))
        root.after(0, lambda: update_output(translated))
        root.after(0, lambda: update_history_dropdown())
        update_status("Ready")
        is_translating = False

    executor.submit(do_translate)

def update_output(text):
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, text)

def copy_output():
    translated_text = output_text.get("1.0", tk.END).strip()
    if translated_text:
        root.clipboard_clear()
        root.clipboard_append(translated_text)
        messagebox.showinfo("Copied", "Translated text copied to clipboard.")
    else:
        messagebox.showwarning("Empty", "No text to copy.")

def speak_output():
    def do_speak():
        translated_text = output_text.get("1.0", tk.END).strip()
        if not translated_text:
            root.after(0, lambda: messagebox.showwarning("Empty", "No text to speak."))
            return
        try:
            engine.say(translated_text)
            engine.runAndWait()
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("TTS Error", str(e)))

    executor.submit(do_speak)

def swap_languages():
    src = source_lang.get()
    tgt = target_lang.get()
    source_lang.set(tgt)
    target_lang.set(src)

def update_history_dropdown():
    history_combo['values'] = [f"{i+1}. {h[0][:20]}..." for i, h in enumerate(history)]

def load_from_history(event):
    selection = history_combo.current()
    if selection >= 0:
        original, translated = history[selection]
        input_text.delete("1.0", tk.END)
        input_text.insert(tk.END, original)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, translated)

def save_translation():
    translated_text = output_text.get("1.0", tk.END).strip()
    if not translated_text:
        messagebox.showwarning("Empty", "No translated text to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            messagebox.showinfo("Saved", f"Translation saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
          

# ------------------- GUI Elements ------------------- #


tk.Label(root, text="Enter text to translate:", bg="#caffe2", font=("Arial", 12)).pack(pady=5)
input_text = tk.Text(root, height=3, width=66)
input_text.pack()

frame = tk.Frame(root, bg="skyblue", bd=4, relief=tk.RIDGE)
frame.pack(pady=15, padx=20)

tk.Label(frame, text="From:", bg="white", fg="black", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=10)
source_lang = ttk.Combobox(frame, values=language_codes, width=15, font=("Arial", 12))
source_lang.set("auto")
source_lang.grid(row=0, column=1, padx=10)

tk.Label(frame, text="To:", bg="white", fg="black", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=10)
target_lang = ttk.Combobox(frame, values=language_codes, width=15, font=("Arial", 12))
target_lang.set("en")
target_lang.grid(row=0, column=3, padx=10)

tk.Button(frame, text="Swap", command=swap_languages, font=("Arial", 12), bg="lightgray").grid(row=0, column=4, padx=10)

auto_detect_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Auto-detect source language", variable=auto_detect_var, bg="#caffe2", font=("Arial", 10)).pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Translate", command=translate_text, font=("Arial", 12), bg="#84D9E8", fg="black").grid(row=0, column=0, padx=5)

tk.Label(root, text="Translated text:", bg="#caffe2", fg="black", font=("Arial", 12)).pack(pady=10)
output_text = tk.Text(root, height=3, width=60, font=("Arial", 12), bg="white", fg="black")
output_text.pack()

action_frame = tk.Frame(root)
action_frame.pack(pady=5)

tk.Button(action_frame, text="Copy Translation", command=copy_output, bg="#84D9E8", font=("Arial", 12)).grid(row=0, column=0, padx=10)
tk.Button(action_frame, text="Speak Translation", command=speak_output, bg="#84D9E8", font=("Arial", 12)).grid(row=0, column=1, padx=10)
tk.Button(action_frame, text="Save Translation", command=save_translation, bg="#84D9E8", font=("Arial", 12)).grid(row=0, column=2, padx=10)

tk.Label(root, text="Translation History:", bg="#caffe2", font=("Arial", 12)).pack(pady=10)
history_combo = ttk.Combobox(root, values=[], font=("Arial", 12), state="readonly", width=58)
history_combo.pack()
history_combo.bind("<<ComboboxSelected>>", load_from_history)

status_var = tk.StringVar(value="Ready.")
tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, bg="#97DAFE", anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()
