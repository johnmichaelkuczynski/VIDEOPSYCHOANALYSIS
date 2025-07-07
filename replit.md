# AI-Powered Personality Insights Platform

## Project Overview
An advanced AI-powered personality insights platform that combines cutting-edge technologies for comprehensive emotional and visual analysis with enhanced cognitive profiling capabilities.

### Key Features
- Multi-modal analysis (images, videos, documents, text)
- Enhanced cognitive profiling with intelligence assessment
- Evidence-based psychological analysis with direct quotations
- Multi-service fallback chains for reliability
- Multiple download formats (PDF, Word, TXT)
- DeepSeek as default AI model with expanded LLM options
- Real-time chat interface for follow-up questions

### Technology Stack
- **Frontend**: React.js with TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Express.js with TypeScript
- **AI Models**: DeepSeek (default), OpenAI GPT-4o, Anthropic Claude, Perplexity
- **Analysis Services**: 
  - Facial Analysis: Azure Face API, Face++, AWS Rekognition
  - Video Analysis: Azure Video Indexer
  - Audio Transcription: Gladia, AssemblyAI, Deepgram, OpenAI Whisper
- **Storage**: In-memory storage with structured data models

## Recent Changes

### DeepSeek Schema Integration Fix (July 7, 2025)
✓ Fixed critical schema validation issue preventing DeepSeek model usage
✓ Updated all Zod schemas to include "deepseek" as valid model option
✓ Changed default model from "openai" to "deepseek" across all schemas
✓ Added DeepSeek to service status display with green indicator
✓ Resolved video upload failures when using DeepSeek model

### Enhanced Cognitive Profiling System (July 7, 2025)
✓ Implemented comprehensive cognitive assessment including:
  - Intelligence level estimation with evidence
  - Cognitive strengths and weaknesses identification
  - Processing style analysis (analytical vs intuitive)
  - Mental agility assessment

✓ Evidence-based analysis system:
  - Direct quotations from speech analysis
  - Specific visual cues and behavioral indicators
  - Observable evidence cited for all assessments

✓ Enhanced analysis prompt structure:
  - Added cognitive_profile section with detailed assessments
  - Improved speech_analysis with vocabulary analysis
  - Added visual_evidence section for facial expressions and body language
  - Enhanced behavioral_indicators for personality trait evidence

### DeepSeek Integration & LLM Expansion (July 7, 2025)
✓ Set DeepSeek as default AI model across all analysis endpoints
✓ Updated fallback chains to prioritize DeepSeek
✓ Enhanced model selection interface in frontend
✓ Maintained compatibility with OpenAI, Anthropic, and Perplexity models

### TXT Export Support (July 7, 2025)
✓ Added comprehensive TXT export functionality
✓ Implemented generateAnalysisTxt function with proper formatting
✓ Updated download route to support txt format
✓ Enhanced frontend with TXT download buttons
✓ Updated API types to include "txt" format option

### Multi-Service Fallback Architecture (Previous)
✓ Implemented robust fallback chains for all external services
✓ Enhanced transcription with multiple service support
✓ Improved facial analysis reliability with service redundancy

## User Preferences
- Default AI Model: DeepSeek (OpenAI-compatible API)
- Analysis Focus: Comprehensive cognitive and psychological profiling
- Evidence Requirement: All assessments must be supported by observable evidence
- Download Formats: PDF, Word, and TXT support required
- Service Reliability: Fallback chains preferred for all external services

## Project Architecture

### Analysis Pipeline
1. **Media Processing**: Images/videos processed with facial analysis and transcription
2. **Multi-Service Analysis**: Fallback chains ensure reliability across all services
3. **Enhanced AI Analysis**: Cognitive profiling with evidence-based reasoning
4. **Document Generation**: Multiple export formats with comprehensive formatting
5. **Real-time Chat**: Follow-up questions and clarifications supported

### API Endpoints
- `/api/upload/media` - Media upload and analysis
- `/api/analyze/text` - Text analysis
- `/api/analyze/document` - Document analysis
- `/api/download/:id` - Multi-format document download
- `/api/chat` - Real-time chat interface
- `/api/share` - Analysis sharing functionality

### Frontend Components
- Media upload interface with drag-and-drop
- Model selection with real-time service status
- Analysis results with formatted display
- Download options (PDF, Word, TXT)
- Chat interface for follow-up questions
- Session management and history

## Development Notes
- Enhanced analysis prompts provide comprehensive cognitive assessments
- Evidence-based reasoning ensures scientific rigor in all analyses
- TXT export maintains formatting and structure for accessibility
- Multi-service architecture provides reliability and redundancy
- DeepSeek integration offers cost-effective high-quality analysis