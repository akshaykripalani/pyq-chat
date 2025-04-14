#!/usr/bin/env python3
"""
PDF File Renaming Script

This script contains mapping dictionaries used to rename PDF files 
by expanding abbreviations for subjects and months into their full forms.
It can process directories of PDF files and rename them according to defined patterns.

Usage:
  python rename_pdfs.py /path/to/base/directory [--dry-run]
  
Options:
  --dry-run    Preview changes without actually renaming files
"""

import os
import sys
import argparse
from pathlib import Path
# Subject mapping dictionary - converts subject code abbreviations to full names
# Used to rename folders or expand subject codes in filenames
subject_mapping = {
    'COA': 'Computer_Organization_And_Architecture',
    'WP': 'Web_Programming',
    'DAA': 'Design_And_Analysis_Of_Algorithms',
    'ML': 'Machine_Learning',
    'AI': 'Artificial_Intelligence',
    'DS': 'Data_Structures',
    'OS': 'Operating_Systems',
    'DBMS': 'Database_Management_Systems',
    'CN': 'Computer_Networks',
    'SE': 'Software_Engineering',
    'TOC': 'Theory_Of_Computation',
    'CG': 'Computer_Graphics',
    'CD': 'Compiler_Design',
    'DM': 'Discrete_Mathematics'
}

# Month mapping dictionary - converts month abbreviations to full month names
# Used to standardize date formats in filenames
month_mapping = {
    'Jan': 'January',
    'Feb': 'February',
    'Mar': 'March',
    'Apr': 'April',
    'May': 'May',  # May is already the full form
    'Jun': 'June',
    'Jul': 'July',
    'Aug': 'August',
    'Sep': 'September',
    'Oct': 'October',
    'Nov': 'November',
    'Dec': 'December'
}\r
\r
def parse_filename(filename):\r
    """\r
    Parse a PDF filename and extract components for renaming.\r
    \r
    Args:\r
        filename (str): Original filename to parse\r
    \r
    Returns:\r
        str: New filename based on parsing rules, or original if parsing fails\r
    """\r
    try:\r
        # Extract components from the filename (this is a simple example)\r
        # Actual implementation would depend on the exact format of your filenames\r
        parts = filename.split('_')\r
        if len(parts) >= 3:\r
            subject_code = parts[0]\r
            month_abbr = parts[1]\r
            \r
            # Convert abbreviations to full forms if they exist in mappings\r
            subject_full = subject_mapping.get(subject_code, subject_code)\r
            month_full = month_mapping.get(month_abbr, month_abbr)\r
            \r
            # Create new filename with expanded abbreviations\r
            new_parts = [subject_full, month_full] + parts[2:]\r
            new_filename = '_'.join(new_parts)\r
            \r
            return new_filename\r
        return filename  # Return original if format doesn't match expected pattern\r
    except Exception as e:\r
        print(f"Error parsing filename '{filename}': {str(e)}")\r
        return filename  # Return original on error\r
\r
def rename_pdf_file(file_path, dry_run=False):\r
    """\r
    Rename a PDF file according to parsing rules.\r
    \r
    Args:\r
        file_path (Path): Path object to the PDF file\r
        dry_run (bool): If True, only show changes without renaming\r
    \r
    Returns:\r
        bool: True if renaming successful or simulated, False on error\r
    """\r
    try:\r
        original_name = file_path.name\r
        new_name = parse_filename(original_name)\r
        \r
        # If the name wouldn't change, skip it\r
        if original_name == new_name:\r
            print(f"  - No change needed for: {original_name}")\r
            return True\r
        \r
        new_path = file_path.parent / new_name\r
        \r
        if dry_run:\r
            print(f"  - Would rename: {original_name} -> {new_name}")\r
        else:\r
            # Perform the actual rename operation\r
            file_path.rename(new_path)\r
            print(f"  - Renamed: {original_name} -> {new_name}")\r
        \r
        return True\r
    except Exception as e:\r
        print(f"  - Error renaming '{file_path.name}': {str(e)}")\r
        return False\r
\r
def process_directories(base_dir, dry_run=False):
    """
    Process all directories in the base directory that match subject codes.
    
    Args:
        base_dir (str): Path to the base directory containing subject folders
    
    Returns:
        int: Number of directories processed
    """
    base_path = Path(base_dir)
    processed_count = 0
    
    print(f"Looking for subject directories in: {base_path}")
    
    # Verify the base directory exists
    if not base_path.exists():
        print(f"Error: Base directory '{base_path}' does not exist")
        return 0
    
    if not base_path.is_dir():
        print(f"Error: '{base_path}' is not a directory")
        return 0
    
    # Get all directories in the base path
    try:
        all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    except PermissionError:
        print(f"Error: Permission denied when accessing '{base_path}'")
        return 0
    except Exception as e:
        print(f"Error accessing '{base_path}': {str(e)}")
        return 0
    
    # Process each directory that matches a subject code
    for dir_path in all_dirs:
        dir_name = dir_path.name\r
        if dir_name in subject_mapping:\r
            processed_count += 1\r
            try:\r
                print(f"\\nProcessing directory: {dir_name} ({subject_mapping[dir_name]})")\r
                \r
                # Get all PDF files in the directory\r
                pdf_files = list(dir_path.glob("*.pdf"))\r
                \r
                if pdf_files:\r
                    print(f"Found {len(pdf_files)} PDF files:")\r
                    renamed_count = 0\r
                    for pdf_file in pdf_files:\r
                        if rename_pdf_file(pdf_file, dry_run):\r
                            renamed_count += 1\r
                    \r
                    if dry_run:\r
                        print(f"Would rename {renamed_count} of {len(pdf_files)} files.")\r
                    else:\r
                        print(f"Successfully renamed {renamed_count} of {len(pdf_files)} files.")\r
                else:\r
                    print("No PDF files found in this directory.")
                    
            except PermissionError:
                print(f"Error: Permission denied when accessing '{dir_path}'")
            except Exception as e:
                print(f"Error processing directory '{dir_path}': {str(e)}")
    
    if processed_count == 0:
        print("No matching subject directories were found.")
    
    return processed_count

def parse_arguments():\r
    """\r
    Parse command line arguments.\r
    \r
    Returns:\r
        argparse.Namespace: Parsed command line arguments\r
    """\r
    parser = argparse.ArgumentParser(\r
        description='PDF File Renaming Tool - Processes PDF files in subject directories.'\r
    )\r
    \r
    parser.add_argument(\r
        'base_directory', \r
        help='Base directory containing subject folders'\r
    )\r
    \r
    parser.add_argument(\r
        '--dry-run',\r
        action='store_true',\r
        help='Preview changes without actually renaming files'\r
    )\r
    \r
    return parser.parse_args()

# Main script execution
if __name__ == "__main__":
    print("PDF renaming tool initialized.")
    print(f"Subject mappings available: {len(subject_mapping)}")
    print(f"Month mappings available: {len(month_mapping)}")
    
    # Parse command line arguments
    try:
        args = parse_arguments()
        
        # Process directories based on the provided base directory\r
        print(f"Running in {'dry-run (preview)' if args.dry_run else 'execution'} mode")\r
        directories_processed = process_directories(args.base_directory, args.dry_run)\r
        \r
        if directories_processed > 0:
            print(f"\nSuccessfully processed {directories_processed} directories.")
        else:
            print("\nNo directories were processed. Please check the path and try again.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
