import pyttsx3
import sys
import multiprocessing

def speak_text(text):
    """Function to run in a separate process to handle TTS"""
    try:
        # Initialize engine within the process to ensure clean state
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"音声生成エラー: {e}")

def main():
    print("読み上げたい文章を入力してください (終了するには Ctrl+C または 'exit' と入力):")

    try:
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            
            try:
                text = input()
            except EOFError:
                break
            
            if text.lower() == 'exit':
                break
            
            if text.strip():
                # Run TTS in a separate process to avoid event loop conflicts
                p = multiprocessing.Process(target=speak_text, args=(text,))
                p.start()
                p.join() # Wait for speech to finish
                
    except KeyboardInterrupt:
        print("\n終了します。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
