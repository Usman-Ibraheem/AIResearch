
import os
import io
import re
import json
from datetime import datetime
from flask import Flask, request, render_template, jsonify, flash, session
from pyngrok import ngrok
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import docx
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import logging
import uuid
from pathlib import Path
import tempfile
import pickle
# Add these imports at the top
import requests
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from html import unescape
import wave
from werkzeug.utils import secure_filename
import speech_recognition as sr
from werkzeug.utils import secure_filename
import pyttsx3
from threading import Thread
from gtts import gTTS
import threading
import openai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Google Generative AI using an environment variable
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("Missing OpenAI API key. Set it as an environment variable.")

openai.api_key = OPENAI_API_KEY
    

# Add this after your existing Flask configuration
TEMP_DIR = Path(tempfile.gettempdir()) / 'paper_analyzer'
TEMP_DIR.mkdir(exist_ok=True)

# Add this after your existing Flask configuration
@app.before_request
def before_request():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    cleanup_old_files()






@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        # Save the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)

            # Open the saved audio file and send to OpenAI API
            with open(temp_audio.name, 'rb') as audio:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )

        # Clean up temporary file
        os.unlink(temp_audio.name)

        return jsonify({'text': transcript})
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Update the save_analysis_data function to ensure proper data serialization
def save_analysis_data(user_id, papers_data, research_paper):
    """Save analysis data to temporary file"""
    try:
        file_path = TEMP_DIR / f"{user_id}_data.pkl"
        data = {
            'papers_data': papers_data,
            'research_paper': research_paper,
            'timestamp': datetime.now()
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Successfully saved data for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis data: {str(e)}")
        return False

def load_analysis_data(user_id):
    """Load analysis data from temporary file"""
    try:
        file_path = TEMP_DIR / f"{user_id}_data.pkl"
        if not file_path.exists():
            return None, None

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Check if data is older than 24 hours
        if (datetime.now() - data['timestamp']).total_seconds() > 86400:
            file_path.unlink(missing_ok=True)
            return None, None

        return data['papers_data'], data['research_paper']
    except Exception as e:
        logger.error(f"Error loading analysis data: {str(e)}")
        return None, None

def cleanup_old_files():
    """Remove files older than 24 hours"""
    try:
        current_time = datetime.now()
        for file_path in TEMP_DIR.glob('*_data.pkl'):
            if (current_time - datetime.fromtimestamp(file_path.stat().st_mtime)).total_seconds() > 86400:
                file_path.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = []

        # Add logging to check text extraction
        logger.info(f"Processing PDF: {pdf_file.filename}")
        logger.info(f"Number of pages: {len(pdf_reader.pages)}")

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text.append(page_text)
                logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters")
            except Exception as e:
                logger.error(f"Error on page {page_num + 1}: {str(e)}")
                continue

        combined_text = "\n".join(text)
        logger.info(f"Total extracted text length: {len(combined_text)}")
        return combined_text
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_file.filename}: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def analyze_paper(text):
    """Analyze research paper text and extract structured information"""
    try:
        # Define the prompt with explicit JSON formatting
        prompt = """
        Analyze this research paper and provide the information in the following JSON format.
        Return ONLY the JSON object without any additional text, markdown formatting, or code blocks:
        {
            "title": "Paper title here",
            "publication_year": "YYYY",
            "publisher": "Publisher name",
            "domain": "Research domain",
            "objective": "Main objective",
            "methods": ["method1", "method2"],
            "technique": "Main technique",
            "advanced_features": ["feature1", "feature2"],
            "solution": "Solution description",
            "results": "Results summary",
            "limitations": "Study limitations",
            "future_work": "Future work",
            "dataset_used": "Dataset description",
            "accuracy_metrics": "Metrics used",
            "keywords": ["keyword1", "keyword2"]
        }
        """

        # Truncate text if too long
        max_chunk_size = 8000
        text_chunk = text[:max_chunk_size]

        # Generate response
        response = model.generate_content(prompt + "\n" + text_chunk)
        response_text = response.text.strip()

        # Clean up the response text
        # Remove any markdown code block indicators
        response_text = response_text.replace('```json', '').replace('```', '')
        response_text = response_text.strip()

        # Make sure the response starts and ends with curly braces
        response_text = response_text.strip()
        if not response_text.startswith('{'):
            response_text = '{' + response_text
        if not response_text.endswith('}'):
            response_text = response_text + '}'

        try:
            # Parse JSON response
            paper_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {str(json_err)}")
            logger.error(f"Problematic JSON text: {response_text}")

            # Attempt to fix common JSON issues
            response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes
            response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas
            response_text = re.sub(r',\s*]', ']', response_text)  # Remove trailing commas in arrays

            try:
                paper_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If still fails, create a basic structure with error information
                paper_data = {
                    "title": "Error Processing Paper",
                    "publication_year": "",
                    "publisher": "",
                    "domain": "",
                    "objective": "Error occurred during analysis",
                    "methods": [],
                    "technique": "",
                    "advanced_features": [],
                    "solution": "",
                    "results": "",
                    "limitations": "Paper could not be properly analyzed",
                    "future_work": "",
                    "dataset_used": "",
                    "accuracy_metrics": "",
                    "keywords": []
                }

        # Ensure all required fields are present
        required_fields = [
            "title", "publication_year", "publisher", "domain", "objective",
            "methods", "technique", "advanced_features", "solution", "results",
            "limitations", "future_work", "dataset_used", "accuracy_metrics", "keywords"
        ]

        for field in required_fields:
            if field not in paper_data:
                paper_data[field] = "" if field not in ["methods", "advanced_features", "keywords"] else []

        # Convert lists if they're strings
        list_fields = ["methods", "advanced_features", "keywords"]
        for field in list_fields:
            if isinstance(paper_data[field], str):
                # Split on commas or semicolons and clean up
                if paper_data[field]:
                    paper_data[field] = [item.strip() for item in re.split(r'[,;]', paper_data[field]) if item.strip()]
                else:
                    paper_data[field] = []

        # Ensure publication_year is a string
        if isinstance(paper_data["publication_year"], int):
            paper_data["publication_year"] = str(paper_data["publication_year"])

        return paper_data

    except Exception as e:
        logger.error(f"Error in analyze_paper: {str(e)}")
        logger.error(f"Response text: {response_text if 'response_text' in locals() else 'No response text'}")

        # Return a basic structure with error information
        return {
            "title": "Error Processing Paper",
            "publication_year": "",
            "publisher": "",
            "domain": "",
            "objective": f"Error occurred during analysis: {str(e)}",
            "methods": [],
            "technique": "",
            "advanced_features": [],
            "solution": "",
            "results": "",
            "limitations": "Paper could not be properly analyzed",
            "future_work": "",
            "dataset_used": "",
            "accuracy_metrics": "",
            "keywords": []
        }

def generate_section_prompt(section_type, papers_data, custom_title=None, previous_sections=None):
    """Creates specific prompts for different paper sections using analyzed papers data"""

    papers_summary = [{
        'title': paper['title'],
        'year': paper['publication_year'],
        'domain': paper['domain'],
        'methods': paper['methods'],
        'results': paper['results'],
        'keywords': paper['keywords']
    } for paper in papers_data]

    base_prompts = {
        "title": f"""Generate a comprehensive academic title for a research paper that synthesizes the following papers:
        {json.dumps(papers_summary, indent=2)}

        {f'Use this title as inspiration: {custom_title}' if custom_title else 'Create an appropriate title.'}
        Return only the title text.""",

        "abstract": f"""Generate a comprehensive academic abstract synthesizing these papers but do not include any results achieved because it is s a review paper:
        {json.dumps(papers_summary, indent=2)}

        Focus on:
        - Key research areas and themes
        - Common methodologies
        - Research implications

        Format as a single paragraph of 250-300 words.""",

        "introduction": f"""Write a detailed academic introduction section with long paragraphs and citations synthesizing:
        {json.dumps(papers_summary, indent=2)}

        Include:
        - Research background and context
        - Problem statement
        - Research gaps identified across papers
        - Significance of the research area
        - Objectives of this literature review

        Do not make subsections for the above mentioned, instead, make them in paragraphs without mentioning the title of the subsection.

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "literature_review": f"""Create a comprehensive literature review synthesizing:
        {json.dumps(papers_summary, indent=2)}

        Organize by:
        - Major themes and research areas
        - Methodological approaches
        - Theoretical frameworks
        - Critical analysis of findings
        - Research gaps

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "methodology_analysis": f"""Analyze the methodological approaches across:
        {json.dumps(papers_summary, indent=2)}

        Focus on:
        - Common research methods
        - Data collection techniques
        - Analysis frameworks
        - Evaluation metrics
        - Methodological strengths and limitations

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "results_discussion": f"""Synthesize and discuss the results from:
        {json.dumps(papers_summary, indent=2)}

        Include:
        - Key findings and patterns
        - Comparative analysis
        - Performance metrics
        - Implementation challenges
        - Practical implications

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "future_directions": f"""Analyze future research directions based on:
        {json.dumps(papers_summary, indent=2)}

        Cover:
        - Emerging trends
        - Research opportunities
        - Technical challenges
        - Potential applications
        - Recommended approaches

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "conclusion": f"""Write a strong conclusion synthesizing:
        {json.dumps(papers_summary, indent=2)}

        Include:
        - Summary of key findings
        - Research contributions
        - Practical implications
        - Limitations
        - Future work recommendations

        {f'Previous sections context: {previous_sections}' if previous_sections else ''}""",

        "references": f"""Generate IEEE format references for:
        {json.dumps(papers_summary, indent=2)}

        Format each reference following IEEE style."""
    }

    return base_prompts.get(section_type, "")

def apply_academic_formatting(text, section_type="general"):
    """Apply academic formatting to text using HTML/CSS styling"""

    # Handle different section types
    if section_type == "heading":
        return f'<h2 class="section-heading">{text}</h2>'

    if section_type == "subheading":
        return f'<h3 class="section-subheading">{text}</h3>'

    if section_type == "title":
        return f'<h1 class="paper-title">{text}</h1>'

    if section_type == "abstract":
        return f'<div class="paper-abstract"><strong>Abstract: </strong>{text}</div>'

    # Process the text content
    formatted_text = text

    # Convert markdown-style headings to HTML
    heading_pattern = r'^#+\s+(.+)$'
    formatted_text = re.sub(heading_pattern, lambda m: f'<h3 class="section-subheading">{m.group(1)}</h3>', formatted_text, flags=re.MULTILINE)

    # Convert markdown-style bullet points to HTML
    bullet_pattern = r'^\*\s+(.+)$'
    formatted_text = re.sub(bullet_pattern, lambda m: f'<li>{m.group(1)}</li>', formatted_text, flags=re.MULTILINE)

    # Wrap bullet points in ul tags
    if re.search(bullet_pattern, text, re.MULTILINE):
        formatted_text = '<ul class="paper-list">' + formatted_text + '</ul>'

    # Convert tables to HTML
    if '|' in formatted_text:
        rows = formatted_text.split('\n')
        table_rows = []
        in_table = False

        for row in rows:
            if '|' in row:
                if not in_table:
                    table_rows.append('<table class="paper-table">')
                    in_table = True

                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                if '-' in row and all(all(c == '-' for c in cell.strip()) for cell in cells):
                    continue

                is_header = not in_table or (len(table_rows) == 2 and all('-' in r for r in rows[1].split('|')))
                cell_tag = 'th' if is_header else 'td'

                formatted_row = '<tr>' + ''.join(f'<{cell_tag}>{cell}</{cell_tag}>' for cell in cells) + '</tr>'
                table_rows.append(formatted_row)
            elif in_table:
                table_rows.append('</table>')
                in_table = False

        if in_table:
            table_rows.append('</table>')

        formatted_text = '\n'.join(table_rows)

    # Convert emphasis markers
    formatted_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_text)
    formatted_text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', formatted_text)

    # Wrap paragraphs
    paragraphs = formatted_text.split('\n\n')
    formatted_paragraphs = []
    for para in paragraphs:
        if not para.startswith(('<h', '<ul', '<table')):
            formatted_paragraphs.append(f'<p class="paper-content">{para}</p>')
        else:
            formatted_paragraphs.append(para)

    formatted_text = '\n'.join(formatted_paragraphs)

    return formatted_text

def generate_academic_table(headers, rows, caption=None):
    """Generate an academically formatted HTML table"""
    table_html = '''
    <div class="academic-table-container" style="margin: 2em 0; overflow-x: auto;">
        <table class="academic-table" style="width: 100%; border-collapse: collapse; font-size: 10pt; margin: 1em 0; border: 1px solid #ddd;">
    '''

    # Add caption if provided
    if caption:
        table_html += f'<caption style="caption-side: top; text-align: left; margin-bottom: 0.5em; font-weight: bold;">{caption}</caption>'

    # Add headers
    table_html += '''
        <thead style="background-color: #f5f5f5;">
            <tr style="border-bottom: 2px solid #666;">
    '''
    for header in headers:
        table_html += f'<th style="padding: 0.8em; text-align: left; font-weight: bold; border: 1px solid #ddd;">{header}</th>'
    table_html += '</tr></thead><tbody>'

    # Add rows
    for row in rows:
        table_html += '<tr style="border-bottom: 1px solid #ddd;">'
        for cell in row:
            table_html += f'<td style="padding: 0.8em; border: 1px solid #ddd;">{cell}</td>'
        table_html += '</tr>'

    table_html += '</tbody></table></div>'
    return table_html

def format_chat_response(response_text, response_type="general"):
    """Format chat responses with appropriate academic styling"""

    if response_type == "table":
        # Convert markdown or plain text table to formatted HTML table
        lines = response_text.strip().split('\n')
        headers = [h.strip() for h in lines[0].strip('|').split('|')]
        rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines[2:]]
        return generate_academic_table(headers, rows)

    elif response_type == "citation":
        return f'<div class="chat-citation" style="font-family: serif; margin: 1em 0; padding-left: 2em; border-left: 3px solid #666;">{response_text}</div>'

    elif response_type == "definition":
        term, definition = response_text.split(':', 1)
        return f'''
        <div class="chat-definition" style="margin: 1em 0; padding: 1em; background-color: #f9f9f9; border-radius: 5px;">
            <strong style="color: #333;">{term}:</strong>
            <span style="display: block; margin-top: 0.5em; padding-left: 1em; border-left: 2px solid #666;">{definition}</span>
        </div>
        '''

    else:
        # Format general chat responses
        formatted_text = response_text

        # Format key academic terms
        academic_terms = {
            r'\b(hypothesis|theory|methodology|analysis|results|conclusion)\b': '<strong>$1</strong>',
            r'\b(p < \d+\.\d+)\b': '<span style="font-family: serif;">$1</span>',
            r'\b(Fig\.|Table|Eq\.)\s*\d+': '<span style="font-family: serif;">$1</span>',
        }

        for pattern, replacement in academic_terms.items():
            formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)

        return f'<div class="chat-response" style="font-family: serif; line-height: 1.6; text-align: justify;">{formatted_text}</div>'

def generate_section_content(prompt, section_type="general"):
    """Generate academically formatted content for research paper sections"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Apply specific formatting based on section type
        if section_type == "title":
            return apply_academic_formatting(content, "title")
        elif section_type == "abstract":
            return apply_academic_formatting(content, "abstract")
        else:
            # Process section content
            formatted_content = []

            # Split content into subsections if they exist
            subsections = content.split('\n\n')

            for subsection in subsections:
                if subsection.startswith('#'):
                    # Handle section headings
                    heading, text = subsection.split('\n', 1)
                    level = len(re.match(r'^#+', heading).group())
                    heading_text = heading.lstrip('#').strip()

                    if level == 1:
                        formatted_content.append(apply_academic_formatting(heading_text, "heading"))
                    else:
                        formatted_content.append(apply_academic_formatting(heading_text, "subheading"))

                    formatted_content.append(apply_academic_formatting(text))
                else:
                    formatted_content.append(apply_academic_formatting(subsection))

            return '\n'.join(formatted_content)

    except Exception as e:
        logger.error(f"Error generating section content: {str(e)}")
        return apply_academic_formatting("Error generating content")

def generate_research_paper(papers_data, custom_title=None):
    """Generate a research paper section by section"""
    try:
        if not papers_data:
            return None

        paper_sections = {}
        previous_sections = {}

        # Generate title first if not provided
        if not custom_title:
            title_content = generate_section_content(
                generate_section_prompt("title", papers_data)
            )
            paper_sections['title'] = title_content
        else:
            paper_sections['title'] = custom_title

        # Define section generation order
        sections = [
            "abstract",
            "introduction",
            "literature_review",
            "methodology_analysis",
            "results_discussion",
            "future_directions",
            "conclusion",
            "references"
        ]

        # Generate each section
        for section in sections:
            logger.info(f"Generating {section} section")

            # Create context from previous sections
            if previous_sections:
                context = "\n\n".join([
                    f"{k.title()}: {v[:200]}..."
                    for k, v in previous_sections.items()
                ])
            else:
                context = None

            # Generate section content
            section_content = generate_section_content(
                generate_section_prompt(
                    section,
                    papers_data,
                    custom_title,
                    context
                )
            )

            paper_sections[section] = section_content
            previous_sections[section] = section_content

            logger.info(f"Completed {section} section")

        # Format the complete paper
        paper_content = f"""
        <div class="paper-header">
            <h1 class="paper-title">{paper_sections['title']}</h1>
            <div class="paper-authors">Generated Literature Review</div>
            <div class="paper-affiliation">Research Analysis System</div>
        </div>

        <div class="paper-abstract">
            <h2>Abstract</h2>
            <div class="paper-content">{paper_sections['abstract']}</div>
        </div>

        <div class="paper-section">
            <h2>I. Introduction</h2>
            <div class="paper-content">{paper_sections['introduction']}</div>
        </div>

        <div class="paper-section">
            <h2>II. Literature Review</h2>
            <div class="paper-content">{paper_sections['literature_review']}</div>
        </div>

        <div class="paper-section">
            <h2>III. Methodology Analysis</h2>
            <div class="paper-content">{paper_sections['methodology_analysis']}</div>
        </div>

        <div class="paper-section">
            <h2>IV. Results and Discussion</h2>
            <div class="paper-content">{paper_sections['results_discussion']}</div>
        </div>

        <div class="paper-section">
            <h2>V. Future Research Directions</h2>
            <div class="paper-content">{paper_sections['future_directions']}</div>
        </div>

        <div class="paper-section">
            <h2>VI. Conclusion</h2>
            <div class="paper-content">{paper_sections['conclusion']}</div>
        </div>

        <div class="paper-references">
            <h2>References</h2>
            <div class="paper-content">{paper_sections['references']}</div>
        </div>
        """

        return paper_content

    except Exception as e:
        logger.error(f"Error generating research paper: {str(e)}")
        return None

def generate_visualizations(papers_data):
    try:
        if not papers_data:
            return {}

        df = pd.DataFrame(papers_data)
        visualizations = {}

        # 1. Publication Trend Over Years
        if 'publication_year' in df.columns and not df['publication_year'].empty:
            years_count = df['publication_year'].value_counts().sort_index()
            fig = px.line(x=years_count.index, y=years_count.values,
                         title='Publications Over Years',
                         labels={'x': 'Year', 'y': 'Number of Publications'})
            fig.update_traces(line_color='#1f77b4')
            visualizations['pub_trend'] = fig.to_html()

        # 2. Domain Distribution
        if 'domain' in df.columns and not df['domain'].empty and df['domain'].notna().any():
            domain_count = df['domain'].value_counts()
            if not domain_count.empty:
                fig = px.pie(values=domain_count.values, names=domain_count.index,
                           title='Research Domains Distribution')
                visualizations['domain_dist'] = fig.to_html()

        # 3. Methods/Techniques Word Cloud-style Visualization
        if 'methods' in df.columns:
            methods = []
            for paper_methods in df['methods']:
                if isinstance(paper_methods, list):
                    methods.extend(paper_methods)
                elif isinstance(paper_methods, str):
                    methods.append(paper_methods)

            if methods:
                methods_count = pd.Series(methods).value_counts()
                if not methods_count.empty:
                    # Create a proper DataFrame for treemap
                    treemap_df = pd.DataFrame({
                        'method': methods_count.index[:15],
                        'count': methods_count.values[:15],
                        'parent': ['Methods'] * len(methods_count.index[:15])
                    })
                    fig = px.treemap(
                        treemap_df,
                        path=['parent', 'method'],
                        values='count',
                        title='Top 15 Methods/Techniques Used'
                    )
                    visualizations['methods_tree'] = fig.to_html()

        # 4. Advanced Features Analysis
        if 'advanced_features' in df.columns:
            features = []
            for paper_features in df['advanced_features']:
                if isinstance(paper_features, list):
                    features.extend(paper_features)
                elif isinstance(paper_features, str):
                    features.append(paper_features)

            if features:
                features_count = pd.Series(features).value_counts()
                if not features_count.empty:
                    fig = px.bar(
                        x=features_count.index[:10],
                        y=features_count.values[:10],
                        title='Top Advanced Features',
                        labels={'x': 'Feature', 'y': 'Frequency'}
                    )
                    visualizations['features_dist'] = fig.to_html()

        # 5. Dataset Usage Analysis
        if 'dataset_used' in df.columns and df['dataset_used'].notna().any():
            dataset_count = df['dataset_used'].value_counts()
            if not dataset_count.empty:
                fig = px.bar(
                    x=dataset_count.index[:8],
                    y=dataset_count.values[:8],
                    title='Common Datasets Used',
                    labels={'x': 'Dataset', 'y': 'Frequency'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                visualizations['dataset_dist'] = fig.to_html()

        # 6. Research Focus Evolution (Keywords)
        if 'keywords' in df.columns:
            keywords = []
            for paper_keywords in df['keywords']:
                if isinstance(paper_keywords, list):
                    keywords.extend(paper_keywords)
                elif isinstance(paper_keywords, str):
                    keywords.append(paper_keywords)

            if keywords:
                keywords_count = pd.Series(keywords).value_counts()
                if not keywords_count.empty:
                    # Create a proper DataFrame for scatter plot
                    scatter_df = pd.DataFrame({
                        'keyword': keywords_count.index[:12],
                        'frequency': keywords_count.values[:12]
                    })
                    fig = px.scatter(
                        scatter_df,
                        x='keyword',
                        y='frequency',
                        size='frequency',
                        title='Research Keywords Distribution',
                        labels={'keyword': 'Keyword', 'frequency': 'Frequency'}
                    )
                    visualizations['keywords_dist'] = fig.to_html()

        return visualizations

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return {}

# Add this function to handle paper searches
def search_open_papers(title, keywords, max_results=10):
    """
    Search for papers using the CORE API
    """
    papers_data = []
    try:
        # Clean and prepare search terms
        cleaned_title = title.strip()
        cleaned_keywords = [k.strip() for k in keywords.split(',') if k.strip()]

        # Combine title and keywords for search
        search_query = cleaned_title
        if cleaned_keywords:
            search_query += " " + " ".join(cleaned_keywords)

        # CORE API configuration
        core_api_url = "https://api.core.ac.uk/v3/search/works"
        headers = {
            "Authorization": "fI1VgyWzxmw9Ci0lAvH4XEoeOY6tGJ8B"
        }
        params = {
            "q": search_query,
            "limit": max_results,
            "fulltext": "true",
            "fields": ["title", "abstract", "authors", "yearPublished", "publisher", "downloadUrl"]
        }

        response = requests.get(core_api_url, headers=headers, params=params, timeout=15)

        if response.status_code == 200:
            core_data = response.json()

            # Check if we have results
            if 'results' not in core_data or not core_data['results']:
                logger.warning("No results found in CORE API response")
                return []

            for paper in core_data['results']:
                # Skip papers without title or abstract
                if not paper.get('title') or not paper.get('abstract'):
                    continue

                # Basic paper data
                paper_data = {
                    "title": paper.get('title', ''),
                    "publication_year": str(paper.get('yearPublished', '')),
                    "publisher": paper.get('publisher', 'Unknown'),
                    "objective": paper.get('abstract', '')[:500],
                    "url": paper.get('downloadUrl', ''),
                    "authors": paper.get('authors', []),
                    "keywords": cleaned_keywords,
                    # Initialize other required fields with default values
                    "domain": "",
                    "methods": [],
                    "technique": "",
                    "results": "",
                    "limitations": "",
                    "dataset_used": "",
                    "accuracy_metrics": "",
                    "advanced_features": [],
                    "solution": "",
                    "future_work": ""
                }

                # Use AI to analyze the abstract
                enrichment_prompt = f"""
                Analyze this research paper abstract and return a JSON object containing key information.

                Title: {paper['title']}
                Abstract: {paper.get('abstract', '')}

                Format your response as a valid JSON object with these exact fields:
                {{
                    "domain": "research domain",
                    "methods": ["method1", "method2"],
                    "technique": "main technique",
                    "results": "key results",
                    "limitations": "main limitations",
                    "dataset_used": "dataset information",
                    "accuracy_metrics": "metrics used",
                    "advanced_features": ["feature1", "feature2"]
                }}
                """

                try:
                    # Get AI response
                    response = model.generate_content(enrichment_prompt)
                    response_text = response.text.strip()

                    # Clean up the response text to ensure valid JSON
                    # Remove any markdown code block indicators
                    response_text = response_text.replace('```json', '').replace('```', '').strip()

                    # Ensure the text starts and ends with curly braces
                    if not response_text.startswith('{'):
                        response_text = '{' + response_text
                    if not response_text.endswith('}'):
                        response_text = response_text + '}'

                    # Try to parse the JSON
                    try:
                        enriched_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If parsing fails, try to fix common JSON issues
                        response_text = response_text.replace("'", '"')  # Replace single quotes
                        response_text = re.sub(r',\s*}', '}', response_text)  # Remove trailing commas
                        response_text = re.sub(r',\s*]', ']', response_text)  # Remove trailing commas in arrays
                        enriched_data = json.loads(response_text)

                    # Update paper data with AI-extracted information
                    # Use .get() with default values to handle missing fields
                    paper_data.update({
                        "domain": enriched_data.get("domain", ""),
                        "methods": enriched_data.get("methods", []),
                        "technique": enriched_data.get("technique", ""),
                        "results": enriched_data.get("results", ""),
                        "limitations": enriched_data.get("limitations", ""),
                        "dataset_used": enriched_data.get("dataset_used", ""),
                        "accuracy_metrics": enriched_data.get("accuracy_metrics", ""),
                        "advanced_features": enriched_data.get("advanced_features", [])
                    })

                    # Ensure list fields are actually lists
                    for field in ["methods", "advanced_features"]:
                        if isinstance(paper_data[field], str):
                            paper_data[field] = [paper_data[field]]
                        elif not isinstance(paper_data[field], list):
                            paper_data[field] = []

                    papers_data.append(paper_data)
                    logger.info(f"Successfully processed paper: {paper['title']}")

                except Exception as e:
                    logger.error(f"Error enriching paper data with AI: {str(e)}")
                    # Still add the paper with basic data even if AI enrichment fails
                    papers_data.append(paper_data)

            logger.info(f"Successfully retrieved and processed {len(papers_data)} papers from CORE API")
            return papers_data
        else:
            logger.error(f"CORE API returned status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return []

    except Exception as e:
        logger.error(f"Unexpected error in paper search: {str(e)}")
        raise Exception(f"An unexpected error occurred during the search: {str(e)}")


# Update the search route to handle the improved search functionality
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        try:
            title = request.form.get('search_title', '').strip()
            keywords = request.form.get('search_keywords', '').strip()
            custom_title = request.form.get('custom_title', '').strip()

            if not title and not keywords:
                flash('Please provide either a title or keywords for search', 'warning')
                return render_template('index.html')

            # Search for papers with progress tracking
            logger.info(f"Starting paper search - Title: {title}, Keywords: {keywords}")
            papers_data = search_open_papers(title, keywords)

            if not papers_data:
                flash('No papers found matching your criteria. Try different keywords or a broader search.', 'warning')
                return render_template('index.html')

            logger.info(f"Found {len(papers_data)} papers")

            # Generate visualizations and research paper
            visualizations = generate_visualizations(papers_data)
            research_paper = generate_research_paper(papers_data, custom_title)

            # Store data for chat functionality
            if 'user_id' not in session:
                session['user_id'] = str(uuid.uuid4())

            if save_analysis_data(session['user_id'], papers_data, research_paper):
                flash(f'Successfully analyzed {len(papers_data)} papers! You can now explore the results and use the chat feature.', 'success')
            else:
                flash('Analysis completed but there was an error saving the data.', 'warning')

            return render_template('index.html',
                                papers_data=papers_data,
                                visualizations=visualizations,
                                research_paper=research_paper)

        except Exception as e:
            logger.error(f"Error in search route: {str(e)}")
            flash(f'An error occurred during the search: {str(e)}', 'danger')
            return render_template('index.html')

    return render_template('index.html')

# Add language selection route
@app.route('/set_language', methods=['POST'])
def set_language():
    try:
        language = request.json.get('language', 'en')
        session['language'] = language
        return jsonify({'status': 'success', 'language': language})
    except Exception as e:
        logger.error(f"Error setting language: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Update your main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            papers_data = []
            files = request.files.getlist('papers')
            custom_title = request.form.get('custom_title', '')

            if not files or files[0].filename == '':
                flash('No files selected', 'warning')
                return render_template('index.html')

            logger.info(f"Processing {len(files)} files")

            for file in files:
                logger.info(f"Processing file: {file.filename}")

                if file.filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif file.filename.endswith('.docx'):
                    text = extract_text_from_docx(file)
                else:
                    logger.warning(f"Unsupported file format: {file.filename}")
                    continue

                if text:
                    paper_data = analyze_paper(text)
                    if paper_data:
                        papers_data.append(paper_data)
                        logger.info(f"Successfully processed {file.filename}")
                    else:
                        logger.error(f"Failed to analyze paper: {file.filename}")
                else:
                    logger.error(f"Failed to extract text from: {file.filename}")

            if not papers_data:
                flash('No valid data could be extracted from the uploaded files', 'warning')
                return render_template('index.html')

            logger.info(f"Successfully processed {len(papers_data)} papers")

            visualizations = generate_visualizations(papers_data)
            research_paper = generate_research_paper(papers_data, custom_title)

            # Store user ID in session (much smaller than storing all data)
            if 'user_id' not in session:
                session['user_id'] = str(uuid.uuid4())

            # Save data to temporary file
            if save_analysis_data(session['user_id'], papers_data, research_paper):
                flash(f'Analysis completed successfully! Processed {len(papers_data)} papers.', 'success')
            else:
                flash('Analysis completed but there was an error saving the data.', 'warning')

            return render_template('index.html',
                                       papers_data=papers_data,
                                       visualizations=visualizations,
                                       research_paper=research_paper)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash(f'An error occurred while processing the files: {str(e)}', 'danger')
            return render_template('index.html')

    return render_template('index.html')

# Update the chat route to properly handle the data
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'response': 'Please provide a message'}), 400

        # Load data using user_id from session
        user_id = session.get('user_id')
        if not user_id:
            logger.error("No user_id found in session")
            return jsonify({'response': 'Session expired. Please upload papers again.'}), 401

        # Load the analysis data
        papers_data, research_paper = load_analysis_data(user_id)
        if not papers_data:
            logger.error(f"No papers data found for user {user_id}")
            return jsonify({'response': 'No analysis data found. Please upload and analyze papers first.'}), 404

        # Generate and format the response
        response = generate_chat_response(user_message, papers_data, research_paper)
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': 'An error occurred while processing your message. Please try again.'
        }), 500

# Modify the generate_chat_response function to handle multiple languages
def generate_chat_response(user_message, papers_data, research_paper):
    try:
        logger.info("Generating chat response")
        logger.debug(f"Number of papers in data: {len(papers_data)}")

        # Get user's preferred language from session
        language = session.get('language', 'en')

        # Prepare papers summary for context
        papers_summary = [{
            'title': paper['title'],
            'objective': paper['objective'],
            'methods': paper['methods'],
            'results': paper['results'],
            'conclusions': paper.get('future_work', '')
        } for paper in papers_data]

        # Create language-specific context
        if language == 'ms':
            context = f"""Anda adalah pembantu penyelidikan akademik yang menganalisis kertas-kertas ini:
            {json.dumps(papers_summary, indent=2)}

            Kandungan kertas penyelidikan:
            {research_paper}

            Analisis kertas-kertas di atas dan berikan respons terperinci untuk soalan ini:
            {user_message}

            Asaskan respons anda hanya pada maklumat dari kertas-kertas yang dianalisis dan kertas penyelidikan yang dihasilkan.
            Sila berikan respons dalam Bahasa Melayu.
            """
        else:
            context = f"""You are an academic research assistant analyzing these papers:
            {json.dumps(papers_summary, indent=2)}

            Research paper content:
            {research_paper}

            Analyze the papers above and provide a detailed response to this question:
            {user_message}

            Base your response only on the information from the analyzed papers and the generated research paper.
            """

        response = model.generate_content(context)
        response_text = response.text.strip()
        logger.info("Successfully generated response")

        # Format the response
        if '|' in response_text and '\n' in response_text:
            return format_chat_response(response_text, "table")
        elif ': ' in response_text and len(response_text.split('\n')) == 1:
            return format_chat_response(response_text, "definition")
        elif any(citation_marker in response_text for citation_marker in ['[', ']', '(', ')']):
            return format_chat_response(response_text, "citation")
        else:
            return format_chat_response(response_text)

    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        error_message = "Maaf, terdapat ralat semasa memproses soalan anda. Sila cuba lagi." if language == 'ms' else "I apologize, but I encountered an error processing your question. Please try again."
        return format_chat_response(error_message)



if __name__ == '__main__':
  app.run(port=10000)
