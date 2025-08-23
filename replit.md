# AI-Powered Personality Insights Platform

## Overview
An advanced AI-powered personality insights platform designed for comprehensive emotional, visual, and cognitive analysis. The platform aims to provide clinical-level psychological evaluations based on multi-modal inputs (images, videos, text). Its vision is to offer deep, evidence-based insights into personality, cognitive patterns, and psychological markers, leveraging cutting-edge AI for professional-grade assessments.

## User Preferences
- Default AI Model: DeepSeek (OpenAI-compatible API)
- Analysis Focus: Comprehensive cognitive and psychological profiling
- Evidence Requirement: All assessments must be supported by observable evidence
- Download Formats: PDF, Word, and TXT support required
- Service Reliability: Fallback chains preferred for all external services

## System Architecture

### UI/UX Decisions
The frontend is built with React.js using TypeScript, Tailwind CSS for styling, and shadcn/ui for components. The design prioritizes a streamlined interface with features like drag-and-drop media upload, real-time segment preview, and interactive chunk selection for document analysis. Analysis results are displayed in a formatted, readable manner, including expandable metrics cards with detailed breakdowns and direct quotations. The layout accommodates multi-modal inputs, providing dedicated sections for image/video analysis, document analysis, and text analysis.

### Technical Implementations
The platform employs a multi-modal analysis approach, integrating visual (images, videos), auditory (transcribed speech), and textual data. Key capabilities include enhanced cognitive profiling with intelligence assessment, evidence-based psychological analysis requiring concrete observable evidence and direct quotations, and multi-service fallback chains for robust operation. Document analysis supports PDF, DOCX, and TXT formats with intelligent chunking and 25 psychological metrics. Video processing includes segment selection and handling of large video files through multipart uploads.

### Feature Specifications
- **Comprehensive Clinical Markers**: Integration of 10 clinical psychological markers for text analysis and 10 visual marker categories for images/videos, ensuring systematic evaluation with observable evidence.
- **Mandatory Demographic & Environmental Anchoring**: All visual analyses begin with a structured assessment of demographic (gender, age, appearance) and environmental details, upon which subsequent psychological insights are anchored.
- **Enhanced Cognitive Profiling**: Includes intelligence level estimation, identification of cognitive strengths/weaknesses, processing style analysis, and mental agility assessment.
- **Advanced Document Analyzer**: Supports upload and analysis of PDF, DOCX, and TXT files, with intelligent chunking (~800 words), interactive chunk selection, and a system for 25 psychological metrics with scoring and detailed explanations.
- **Speech/Text Content Integration**: Prioritizes content analysis, discussing what people say/write, enhancing cognitive profiling based on vocabulary and reasoning patterns, and increasing direct quotations.
- **Download System**: Supports reliable TXT downloads, with previous PDF and Word functionalities streamlined to ensure consistency.

### System Design Choices
The backend uses Express.js with TypeScript. The analysis pipeline involves media processing (facial analysis, transcription), multi-service AI analysis with fallback chains, and document generation in various export formats. An in-memory storage solution is used for structured data models. API endpoints include media upload (`/api/upload/media`, `/api/upload/media-multipart`), text analysis (`/api/analyze/text`), document download (`/api/download/:id`), and a real-time chat interface (`/api/chat`). The system is designed for high reliability and redundancy through its multi-service architecture and robust error handling.

## External Dependencies

- **AI Models**: DeepSeek (default), OpenAI GPT-4o, Anthropic Claude, Perplexity
- **Facial Analysis Services**: Azure Face API, Face++, AWS Rekognition
- **Video Analysis Services**: Azure Video Indexer
- **Audio Transcription Services**: Gladia, AssemblyAI, Deepgram, OpenAI Whisper
- **Document Processing**: Python document parser integration (for advanced PDF processing)