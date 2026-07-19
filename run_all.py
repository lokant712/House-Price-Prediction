import subprocess
import sys
import os
import time

def run_step(command, cwd=None, description=""):
    print(f"\n>>> Running step: {description}...")
    print(f"Executing: {' '.join(command)}")
    start_time = time.time()
    
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds.")
    
    if result.returncode != 0:
        print(f"Error executing step: {description}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(f"Success! {description} completed.")
        if result.stdout.strip():
            print("Output snippet:")
            print("\n".join(result.stdout.strip().split("\n")[:10]))
            if len(result.stdout.strip().split("\n")) > 10:
                print("...")

def main():
    print("==================================================")
    print("HOUSE PRICE PREDICTION - REPRODUCIBILITY RUNNER")
    print("==================================================")
    
    # Set CWD to script directory if run from elsewhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Run the Machine Learning Pipeline
    run_step(
        command=[sys.executable, "src/run_pipeline.py"],
        cwd=script_dir,
        description="ML Pipeline Execution (Pre-processing, training, tuning, visualising)"
    )
    
    # 2. Run Report Compilation
    run_step(
        command=[sys.executable, "src/generate_report.py"],
        cwd=script_dir,
        description="DOCX Report Generation"
    )
    
    # 3. Convert Report to PDF
    print("\n>>> Converting DOCX report to PDF using docx2pdf...")
    try:
        from docx2pdf import convert
        convert("reports/report.docx", "reports/report.pdf")
        print("Success! PDF Report generated: reports/report.pdf")
    except Exception as e:
        print("Failed to convert DOCX to PDF using docx2pdf. Details:")
        print(e)
        print("Falling back to MS Word win32com interface...")
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc_path = os.path.abspath(os.path.join(script_dir, "reports/report.docx"))
            pdf_path = os.path.abspath(os.path.join(script_dir, "reports/report.pdf"))
            doc = word.Documents.Open(doc_path)
            # wdFormatPDF = 17
            doc.SaveAs(pdf_path, FileFormat=17)
            doc.Close()
            word.Quit()
            print("Success! PDF Report generated via COM fallback: reports/report.pdf")
        except Exception as e2:
            print("Failed to convert PDF via COM fallback. Details:")
            print(e2)
            print("Please ensure Microsoft Word is installed on your system to compile reports/report.pdf.")
            sys.exit(1)
            
    print("\n==================================================")
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == '__main__':
    main()
