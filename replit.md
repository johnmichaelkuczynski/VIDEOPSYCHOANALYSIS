# AI-Powered Personality Insights Platform

## Project Overview
An advanced AI-powered personality insights platform that combines cutting-edge technologies for comprehensive emotional and visual analysis with enhanced cognitive profiling capabilities.

### Key Features
- Multi-modal analysis (images, videos, text)
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

### ENHANCED ANALYSIS METHODOLOGY: Mandatory Demographic & Environmental Anchoring (August 20, 2025)
✓ **IMPLEMENTED STRUCTURED ANALYSIS OPENING**: All image and video analyses now begin with mandatory demographic and environmental observations
✓ Updated analysis prompts to require opening assessment of: gender, age, appearance, body posture, body language, setting, background
✓ Enhanced psychological analysis anchoring - all subsequent statements must reference and build upon demographic/environmental observations
✓ Applied to both image analysis and video segment analysis endpoints for consistency
✓ Ensures comprehensive observational foundation before psychological assessment begins
✓ Improves analysis depth and evidence-based reasoning by grounding insights in observable elements

### Analysis Structure Enhancement:
- **Opening Requirements**: Gender, age range, physical appearance, body posture, body language, setting details, environmental context
- **Anchored Assessment**: All psychological insights must reference and build upon demographic/environmental observations
- **Evidence-Based Foundation**: Observable elements provide concrete grounding for personality assessments
- **Consistent Methodology**: Applied across all visual media analysis (images, videos)

### CRITICAL SUCCESS: Image Analysis System Fully Restored (July 30, 2025)
✓ **COMPLETELY FIXED IMAGE ANALYSIS**: Resolved all critical errors that were preventing comprehensive psychological analysis
✓ Fixed "buffer is not defined" error that was causing facial analysis system to crash
✓ Restored AWS Rekognition facial detection with proper emotional and demographic analysis
✓ Implemented comprehensive 9-parameter psychoanalytic framework for image analysis
✓ Enhanced analysis to extract affect regulation, defensive structure, attachment signals, cognitive processing
✓ Fixed variable initialization order to ensure facial data is processed before analysis creation
✓ Analysis now provides detailed insights based on actual facial expressions and emotional data
✓ Successfully tested with real user images showing detailed psychological assessment
✓ Export system properly includes comprehensive image analysis in all formats (PDF, DOCX, TXT)
✓ Formatting cleaned to remove all markdown symbols from display and downloads

### CRITICAL FIX: Large Video Upload Support with Multipart Handling (July 29, 2025)
✓ **RESOLVED 413 PAYLOAD TOO LARGE ERROR**: Fixed critical issue preventing upload of large videos (80MB+)
✓ Implemented proper multipart/form-data upload system to replace inefficient JSON base64 encoding
✓ Added new `/api/upload/media-multipart` endpoint specifically for large file handling
✓ Updated frontend to automatically detect videos and use FormData upload method
✓ Enhanced video segment analysis with real AI processing instead of placeholder content
✓ Fixed video segment selection endpoint communication between frontend and backend
✓ Added comprehensive error handling and proper temp file management for large videos
✓ Integrated facial analysis, audio transcription, and AI-powered personality assessment
✓ Enhanced segment analysis with DeepSeek, Anthropic Claude, and OpenAI model support

### Large Video Processing Features:
- **File Size Support**: Now handles videos over 80MB through multipart uploads
- **Automatic Routing**: Frontend automatically uses appropriate upload method based on file type
- **Real AI Analysis**: Video segments processed with facial analysis, audio transcription, and personality insights
- **Multi-Service Support**: Fallback chains ensure reliability across all AI and analysis services
- **Proper Cleanup**: Temporary files managed efficiently to prevent storage bloat

### COMPLETE REBUILD: Advanced Document Analysis with 25 Psychological Metrics (July 10, 2025)
✓ **ADVANCED DOCUMENT ANALYZER**: Implemented comprehensive document analysis system per user specification
✓ Added document upload endpoints with robust PDF/DOCX/TXT parsing
✓ Implemented intelligent document chunking system (~800 words per chunk)
✓ Created interactive chunk selection UI with checkboxes and previews
✓ Built 25 psychological metrics analysis system with scoring (0-100)
✓ Added expandable metrics cards with detailed analysis and quotes
✓ Implemented formatting preservation for PDF/DOCX documents
✓ Added regenerate and refine analysis capabilities
✓ Enhanced UI with 3-column layout (Image/Video + Document + Text)
✓ Integrated document analysis with existing export system (PDF/DOCX/TXT)
✓ Added Python document parser integration for advanced PDF processing
✓ Created sophisticated metrics display with progress bars and detailed breakdowns
✓ Implemented chunk-based analysis workflow for optimal processing
✓ Added comprehensive error handling and user feedback

### Document Analysis Features:
- **File Support**: PDF, DOCX, TXT with formatting preservation
- **Chunking System**: Intelligent ~800-word chunks with logical paragraph breaks
- **25 Metrics**: Complete psychological profiling with scores and explanations
- **Interactive UI**: Expandable cards, chunk selection, progress tracking
- **Export Integration**: All formats support document analysis results
- **Regeneration**: Users can refine and regenerate analysis with same chunks

### Video Segment Selection System Implementation (July 7, 2025)
✓ Successfully implemented complete video chunking system with user-selectable 3-second segments
✓ Added automatic video duration detection and segment validation
✓ Enhanced UI with real-time segment preview and processing time expectations
✓ Fixed PayloadTooLargeError with 50MB file size limits and early validation
✓ Added 10-minute server timeout for complex video processing
✓ Tested and verified: 6.74s video properly processed with 0-3s segment selection
✓ Confirmed multi-service fallback chains working (Face++, AssemblyAI transcription)

### Enhanced Speech/Text Content Integration (July 7, 2025)
✓ Dramatically enhanced analysis prompts to prioritize speech and text content
✓ Added comprehensive content analysis that discusses what people actually say/write
✓ Enhanced cognitive profiling based on vocabulary, reasoning patterns, and idea sophistication
✓ Increased direct quotations from 3-5 to 5-8 meaningful examples
✓ Added content themes analysis revealing interests, expertise, and priorities
✓ Enhanced personality insights based on topics discussed and communication style
✓ Improved character and values assessment through expressed ideas and perspectives

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