# AI-Powered Personality Insights Platform

## Overview
The AI-Powered Personality Insights Platform is an advanced system designed to provide comprehensive emotional and visual analysis, combined with enhanced cognitive profiling. Its primary purpose is to offer deep, evidence-based psychological evaluations from various data inputs (images, videos, text). The platform aims to be a robust tool for detailed human behavioral and cognitive assessment, moving beyond basic personality insights to clinical-level evaluations.

Key capabilities include multi-modal analysis, sophisticated cognitive profiling, and the ability to cite direct quotations as evidence. The system prioritizes reliability through multi-service fallback chains and offers flexible output formats. The vision is to provide a highly accurate and dependable solution for professional psychological assessment, leveraging cutting-edge AI models like DeepSeek.

## User Preferences
- Default AI Model: DeepSeek (OpenAI-compatible API)
- Analysis Focus: Comprehensive cognitive and psychological profiling
- Evidence Requirement: All assessments must be supported by observable evidence
- Download Formats: TXT format only for reliable export
- Service Reliability: Fallback chains preferred for all external services

## System Architecture

### Core Design Principles
The platform is built on a multi-modal analysis framework, integrating various data types for a holistic assessment. A key architectural decision is the implementation of multi-service fallback chains to ensure high reliability and continuous operation, even if one external service fails. The system emphasizes evidence-based analysis, requiring all AI-generated insights to be anchored in observable data and direct quotations.

### Technical Implementation Details
- **Frontend**: Developed with React.js, TypeScript, Tailwind CSS, and shadcn/ui for a modern and responsive user interface. The UI includes components for media upload, model selection, analysis results display, download options, and a real-time chat interface.
- **Backend**: Implemented using Express.js with TypeScript, managing API endpoints for media processing, text analysis, document generation, and real-time chat. It handles large file uploads via multipart/form-data.
- **Analysis Methodology**:
    - **Clinical Psychological Evaluation**: Integrated comprehensive clinical markers into all analysis types. Text analysis includes 10 markers (e.g., affect in language, cognitive organization), and visual analysis includes 10 categories (e.g., facial expressions, posture). Every assessment requires concrete observable evidence and specific examples.
    - **Mandatory Demographic & Environmental Anchoring**: All image and video analyses begin with mandatory observations of gender, age range, physical appearance, body posture, body language, setting, and background. Subsequent psychological insights must build upon these initial observations.
    - **Cognitive Profiling**: Includes estimation of intelligence level, identification of cognitive strengths/weaknesses, processing style analysis, and mental agility assessment, all supported by evidence.
    - **Speech/Text Content Integration**: Analysis prompts prioritize speech and text content, focusing on vocabulary, reasoning patterns, idea sophistication, and content themes, with a focus on extracting 5-8 meaningful direct quotations.
- **Data Handling**: In-memory storage is used with structured data models. Large video files are handled through multipart uploads, and temporary files are managed efficiently.
- **Document Generation**: Supports TXT format for analysis downloads.
- **UI/UX**: Features a streamlined interface for downloads, interactive chunk selection for document analysis, and a 3-column layout (Image/Video + Document + Text) for enhanced user experience.

### Key Features Implemented
- Multi-modal analysis (images, videos, text).
- Enhanced cognitive profiling with intelligence assessment.
- Evidence-based psychological analysis with direct quotations.
- Multi-service fallback chains for reliability.
- TXT download format.
- DeepSeek as default AI model with expanded LLM options.
- Real-time chat interface for follow-up questions.
- Comprehensive clinical psychological evaluation integration for both text and visual analysis.
- Robust image analysis system.
- Support for large video uploads via multipart handling.
- Advanced document analysis system with intelligent chunking and 25 psychological metrics.
- Video segment selection system with user-selectable 3-second segments.

## External Dependencies
- **AI Models**: DeepSeek (default), OpenAI GPT-4o, Anthropic Claude, Perplexity.
- **Facial Analysis Services**: Azure Face API, Face++, AWS Rekognition.
- **Video Analysis Services**: Azure Video Indexer.
- **Audio Transcription Services**: Gladia, AssemblyAI, Deepgram, OpenAI Whisper.
- **Document Parsing**: Python document parser integration (for advanced PDF processing).