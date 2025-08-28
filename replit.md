# AI-Powered Personality Insights Platform

## Overview
This project is an advanced AI-powered personality insights platform designed for comprehensive emotional, visual, and cognitive analysis. Its purpose is to provide highly detailed, evidence-based psychological evaluations from various media inputs (images, videos, text). The platform aims to deliver professional-grade insights, moving beyond basic personality assessments to offer clinical-level psychological evaluations with documented evidence and direct quotations. It supports multi-modal analysis, incorporates enhanced cognitive profiling, and offers robust reliability through multi-service fallback chains, with results exportable in multiple formats.

## Recent Changes (August 2025)
- **ZHI 1 (Anthropic) Fully Restored**: Added ZHI 1 back to all analysis types including video analysis with comprehensive 60-question framework
- **Comprehensive 60-Question Analysis**: Implemented complete structured analysis with all 60 personality questions (1-20 core + 40 deeper insights) plus visual/textual markers for all models
- **Provider-Specific Analysis Prompts**: Anthropic uses transcript-based analysis, while other models use direct image/video analysis, with identical comprehensive question coverage
- **Structured JSON Output**: All models return standardized JSON with core_questions, personality_40_60, visual_markers, and textual_markers sections
- **ZHI 1 Service Status Integration**: Added ZHI 1 to all service status lists and model selection dropdowns with proper status indicators
- **Custom Video Duration Selector**: Added user-selectable video segment duration (1-10 seconds) for both large and small video uploads with intuitive slider interface
- **Video Analysis Enhancement**: Updated video segment duration from 5 to 10 seconds and corrected upload limits to 30 seconds maximum
- **Visual Analysis Accuracy**: Added critical color precision instructions to prevent clothing description errors and improve observational accuracy
- **Video Upload Stability**: Fixed video upload crashes for files >30 seconds with comprehensive error handling, timeout protection (30s), file size limits (500MB), and automatic cleanup
- **Consolidated Downloads**: Added comprehensive download functionality that combines all analysis modules (psychological Q1-Q18, intelligence Q19-Q36, 40 parameters, clinical assessments) into single documents
- **Bulk Text Operations**: Implemented bulk select/deselect for document text chunks to handle large documents efficiently (tested with 72+ chunks)
- **User Experience**: Added clear video length warnings in upload interface and improved error messages throughout upload pipeline
- **JSON Error Handling**: Robust JSON parsing error handling prevents app crashes during video processing

## User Preferences
- Default AI Model: DeepSeek (OpenAI-compatible API)
- Analysis Focus: Comprehensive cognitive and psychological profiling
- Evidence Requirement: All assessments must be supported by observable evidence
- Download Formats: PDF, Word, and TXT support required
- Service Reliability: Fallback chains preferred for all external services

## System Architecture

### UI/UX Decisions
The frontend is built with React.js, TypeScript, Tailwind CSS, and shadcn/ui, featuring a modern and responsive design. Key UI elements include:
- A drag-and-drop media upload interface.
- An interactive chunk selection UI for documents with checkboxes and previews.
- Expandable metrics cards with detailed analysis and quotes.
- A 3-column layout for integrated Image/Video, Document, and Text analysis.
- Real-time service status indicators for AI models.
- Download options for various formats and a real-time chat interface.
- UI improvements include fixed scrolling for popups, auto-selection of document chunks, and optimized layout for protocol questions.

### Technical Implementations
The platform utilizes a robust technical stack:
- **Frontend**: React.js, TypeScript, Tailwind CSS, shadcn/ui.
- **Backend**: Express.js with TypeScript, handling API endpoints for uploads, analysis, downloads, and chat.
- **Analysis Pipeline**:
    1.  **Media Processing**: Images and videos are processed for facial analysis and audio transcription.
    2.  **Multi-Service Analysis**: Robust fallback chains ensure reliability across all external services.
    3.  **Enhanced AI Analysis**: Utilizes advanced AI models for comprehensive cognitive and psychological profiling with evidence-based reasoning.
    4.  **Document Generation**: Supports multiple export formats (PDF, Word, TXT) with comprehensive formatting.
- **Core Functionality**:
    -   **Document Analysis**: Advanced system for PDF, DOCX, TXT with intelligent chunking (~800 words), 25 psychological metrics with scoring (0-100), and formatting preservation.
    -   **Video Processing**: Supports large video uploads (80MB+) via multipart/form-data, with automatic segment selection (3-second chunks) and real-time processing.
    -   **Cognitive Profiling**: Integrated comprehensive assessment including intelligence estimation, cognitive strengths/weaknesses, processing style, and mental agility, all supported by direct evidence.
    -   **Clinical Psychological Evaluation**: Integrates 10 clinical psychological markers for text and 10 for visual analysis, ensuring systematic assessment with observable evidence and direct quotations.
    -   **Demographic & Environmental Anchoring**: All image and video analyses start with mandatory demographic and environmental observations to ground psychological insights.
    -   **Speech/Text Content Integration**: Enhanced analysis prompts prioritize and analyze speech/text content, focusing on vocabulary, reasoning patterns, and content themes.

### System Design Choices
- **Modularity**: Separation of frontend and backend components allows for independent development and scaling.
- **Reliability**: Multi-service fallback chains are implemented for all external dependencies (facial analysis, transcription, AI models) to ensure continuous operation and data integrity.
- **Scalability**: Designed to handle large files (e.g., videos) efficiently through multipart uploads and optimized processing.
- **Evidence-Based Design**: A core principle is that all AI-generated insights must be backed by observable evidence, direct quotations, or specific behavioral indicators from the source material.
- **Structured Output**: Analysis outputs are structured to provide detailed, scorable metrics and expandable explanations.

## External Dependencies
- **AI Models**:
    -   DeepSeek (default)
    -   OpenAI GPT-4o
    -   Anthropic Claude
    -   Perplexity
-   **Facial Analysis**:
    -   Azure Face API
    -   Face++
    -   AWS Rekognition
-   **Video Analysis**:
    -   Azure Video Indexer
-   **Audio Transcription**:
    -   Gladia
    -   AssemblyAI
    -   Deepgram
    -   OpenAI Whisper
-   **Storage**: In-memory storage with structured data models.
-   **Document Parsing**: Python document parser for advanced PDF processing.