import os
from PyPDF2 import PdfMerger

def concatenate_pdfs(directory):
    """
    Concatenate all PDF files in the given directory into a single PDF file.
    Files are merged in alphabetical order.
    """
    # Create a PDF merger object
    merger = PdfMerger()
    
    # Get all PDF files from the directory
    pdf_files = [f for f in os.listdir(directory) 
                 if f.lower().endswith('.pdf')]
    
    # Sort files alphabetically
    pdf_files.sort()
    
    # Print found PDF files
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"- {pdf}")
    
    # Merge PDFs
    try:
        for pdf in pdf_files:
            file_path = os.path.join(directory, pdf)
            merger.append(file_path)
            print(f"Added: {pdf}")
        
        # Write the merged PDF to a file
        output_path = os.path.join(directory, "merged_document.pdf")
        merger.write(output_path)
        merger.close()
        
        print(f"\nSuccess! Merged PDF saved as: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        merger.close()

# Directory path
directory = "DDNEA"

# Run the concatenation
if __name__ == "__main__":
    if os.path.exists(directory):
        concatenate_pdfs(directory)
    else:
        print(f"Directory '{directory}' not found!")