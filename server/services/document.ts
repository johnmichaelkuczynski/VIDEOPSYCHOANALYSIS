import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, BorderStyle } from 'docx';
import { Analysis } from '../../shared/schema';
import puppeteer from 'puppeteer';

const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// Function to generate plain text format
export function generateAnalysisTxt(analysis: Analysis): string {
  const personalityInsights = analysis.personalityInsights as any || {};
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  let txtContent = '';
  
  // Header
  txtContent += '='.repeat(80) + '\n';
  txtContent += 'AI-POWERED PSYCHOLOGICAL ANALYSIS REPORT\n';
  txtContent += '='.repeat(80) + '\n\n';
  
  txtContent += `Analysis ID: ${analysis.id}\n`;
  txtContent += `Created: ${analysis.createdAt ? new Date(analysis.createdAt).toLocaleString() : 'Unknown'}\n`;
  txtContent += `Media Type: ${analysis.mediaType}\n`;
  txtContent += `People Detected: ${personalityInsights.peopleCount || 1}\n\n`;
  
  // Check for video analysis data
  if (personalityInsights.videoAnalysis && personalityInsights.videoAnalysis.analysisText) {
    txtContent += 'COMPREHENSIVE PSYCHOANALYTIC ASSESSMENT:\n';
    txtContent += '='.repeat(60) + '\n\n';
    // Clean up analysis text by removing markdown formatting
    const cleanAnalysisText = personalityInsights.videoAnalysis.analysisText
      .replace(/#{1,6}\s*/g, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove code blocks
      .replace(/^\s*[-*+]\s+/gm, '• ') // Convert bullets to simple format
      .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered lists
      .trim();
    txtContent += cleanAnalysisText + '\n\n';
  } else if (personalityInsights.comprehensiveAnalysis) {
    // Check for image analysis data
    txtContent += 'COMPREHENSIVE PSYCHOANALYTIC ASSESSMENT:\n';
    txtContent += '='.repeat(60) + '\n\n';
    // Clean up analysis text by removing markdown formatting
    const analysisText = typeof personalityInsights.comprehensiveAnalysis === 'string' 
      ? personalityInsights.comprehensiveAnalysis 
      : JSON.stringify(personalityInsights.comprehensiveAnalysis);
    const cleanAnalysisText = analysisText
      .replace(/#{1,6}\s*/g, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove code blocks
      .replace(/^\s*[-*+]\s+/gm, '• ') // Convert bullets to simple format
      .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered lists
      .trim();
    txtContent += cleanAnalysisText + '\n\n';
    
    if (personalityInsights.faceAnalysis) {
      txtContent += 'FACIAL ANALYSIS DATA:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += JSON.stringify(personalityInsights.faceAnalysis, null, 2) + '\n\n';
    }
    
    if (personalityInsights.videoAnalysis?.segmentInfo) {
      txtContent += 'SEGMENT INFORMATION:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += `Analyzed Segment: ${personalityInsights.videoAnalysis.segmentInfo.label}\n`;
      txtContent += `Duration: ${personalityInsights.videoAnalysis.segmentInfo.duration} seconds\n`;
      txtContent += `Processing Time: ${personalityInsights.videoAnalysis.processingTime}\n`;
      txtContent += `AI Model: ${personalityInsights.videoAnalysis.model}\n\n`;
    }
    
    if (personalityInsights.videoAnalysis?.audioTranscription?.transcription) {
      txtContent += 'AUDIO TRANSCRIPTION:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += `"${personalityInsights.videoAnalysis.audioTranscription.transcription}"\n\n`;
    }
  } else if (isMultiPersonAnalysis) {
    txtContent += personalityInsights.overviewSummary + '\n\n';
    
    personalityInsights.individualProfiles.forEach((profile: any, index: number) => {
      txtContent += '='.repeat(60) + '\n';
      txtContent += `INDIVIDUAL PROFILE ${index + 1}\n`;
      txtContent += '='.repeat(60) + '\n\n';
      
      txtContent += `SUMMARY:\n${profile.summary || 'No summary available'}\n\n`;
      
      const detailed = profile.detailed_analysis || {};
      
      if (detailed.cognitive_profile) {
        txtContent += 'COGNITIVE PROFILE:\n';
        txtContent += '-'.repeat(40) + '\n';
        if (detailed.cognitive_profile.intelligence_assessment) {
          txtContent += `Intelligence Assessment: ${detailed.cognitive_profile.intelligence_assessment}\n\n`;
        }
        if (detailed.cognitive_profile.cognitive_strengths) {
          txtContent += `Cognitive Strengths:\n`;
          detailed.cognitive_profile.cognitive_strengths.forEach((strength: string) => {
            txtContent += `• ${strength}\n`;
          });
          txtContent += '\n';
        }
        if (detailed.cognitive_profile.cognitive_weaknesses) {
          txtContent += `Cognitive Weaknesses:\n`;
          detailed.cognitive_profile.cognitive_weaknesses.forEach((weakness: string) => {
            txtContent += `• ${weakness}\n`;
          });
          txtContent += '\n';
        }
        if (detailed.cognitive_profile.processing_style) {
          txtContent += `Processing Style: ${detailed.cognitive_profile.processing_style}\n\n`;
        }
        if (detailed.cognitive_profile.mental_agility) {
          txtContent += `Mental Agility: ${detailed.cognitive_profile.mental_agility}\n\n`;
        }
      }
      
      if (detailed.personality_core) {
        txtContent += `PERSONALITY CORE:\n${detailed.personality_core}\n\n`;
      }
      
      if (detailed.thought_patterns) {
        txtContent += `THOUGHT PATTERNS:\n${detailed.thought_patterns}\n\n`;
      }
      
      if (detailed.emotional_intelligence) {
        txtContent += `EMOTIONAL INTELLIGENCE:\n${detailed.emotional_intelligence}\n\n`;
      }
      
      if (detailed.speech_analysis) {
        txtContent += 'SPEECH ANALYSIS:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.speech_analysis.key_quotes && detailed.speech_analysis.key_quotes.length > 0) {
          txtContent += 'Key Quotes:\n';
          detailed.speech_analysis.key_quotes.forEach((quote: string, i: number) => {
            txtContent += `${i + 1}. "${quote}"\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.speech_analysis.vocabulary_analysis) {
          txtContent += `Vocabulary Analysis: ${detailed.speech_analysis.vocabulary_analysis}\n\n`;
        }
        
        if (detailed.speech_analysis.speech_patterns) {
          txtContent += `Speech Patterns: ${detailed.speech_analysis.speech_patterns}\n\n`;
        }
        
        if (detailed.speech_analysis.emotional_tone) {
          txtContent += `Emotional Tone: ${detailed.speech_analysis.emotional_tone}\n\n`;
        }
      }
      
      if (detailed.visual_evidence) {
        txtContent += 'VISUAL EVIDENCE:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.visual_evidence.facial_expressions) {
          txtContent += `Facial Expressions: ${detailed.visual_evidence.facial_expressions}\n\n`;
        }
        
        if (detailed.visual_evidence.body_language) {
          txtContent += `Body Language: ${detailed.visual_evidence.body_language}\n\n`;
        }
        
        if (detailed.visual_evidence.emotional_indicators) {
          txtContent += `Emotional Indicators: ${detailed.visual_evidence.emotional_indicators}\n\n`;
        }
      }
      
      if (detailed.professional_insights) {
        txtContent += `PROFESSIONAL INSIGHTS:\n${detailed.professional_insights}\n\n`;
      }
      
      if (detailed.relationships) {
        txtContent += 'RELATIONSHIPS:\n';
        txtContent += '-'.repeat(40) + '\n';
        txtContent += `Current Status: ${detailed.relationships.current_status || 'Not specified'}\n`;
        txtContent += `Parental Status: ${detailed.relationships.parental_status || 'Not specified'}\n`;
        txtContent += `Ideal Partner: ${detailed.relationships.ideal_partner || 'Not specified'}\n\n`;
      }
      
      if (detailed.growth_areas) {
        txtContent += 'GROWTH AREAS:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.growth_areas.strengths) {
          txtContent += 'Strengths:\n';
          detailed.growth_areas.strengths.forEach((strength: string) => {
            txtContent += `• ${strength}\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.growth_areas.challenges) {
          txtContent += 'Challenges:\n';
          detailed.growth_areas.challenges.forEach((challenge: string) => {
            txtContent += `• ${challenge}\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.growth_areas.development_path) {
          txtContent += `Development Path: ${detailed.growth_areas.development_path}\n\n`;
        }
      }
    });
  } else {
    // Single person analysis (including video analysis)
    const profile = personalityInsights.individualProfiles?.[0] || personalityInsights;
    
    // Handle non-video analysis data  
    if (!personalityInsights.videoAnalysis) {
      txtContent += `SUMMARY:\n${profile.summary || 'No summary available'}\n\n`;
    }
    
    const detailed = profile.detailed_analysis || {};
    
    if (detailed.cognitive_profile) {
      txtContent += 'COGNITIVE PROFILE:\n';
      txtContent += '-'.repeat(40) + '\n';
      if (detailed.cognitive_profile.intelligence_assessment) {
        txtContent += `Intelligence Assessment: ${detailed.cognitive_profile.intelligence_assessment}\n\n`;
      }
      if (detailed.cognitive_profile.cognitive_strengths) {
        txtContent += `Cognitive Strengths:\n`;
        detailed.cognitive_profile.cognitive_strengths.forEach((strength: string) => {
          txtContent += `• ${strength}\n`;
        });
        txtContent += '\n';
      }
      if (detailed.cognitive_profile.cognitive_weaknesses) {
        txtContent += `Cognitive Weaknesses:\n`;
        detailed.cognitive_profile.cognitive_weaknesses.forEach((weakness: string) => {
          txtContent += `• ${weakness}\n`;
        });
        txtContent += '\n';
      }
      if (detailed.cognitive_profile.processing_style) {
        txtContent += `Processing Style: ${detailed.cognitive_profile.processing_style}\n\n`;
      }
      if (detailed.cognitive_profile.mental_agility) {
        txtContent += `Mental Agility: ${detailed.cognitive_profile.mental_agility}\n\n`;
      }
    }
    
    // Continue with other sections like personality_core, thought_patterns, etc.
    if (detailed.personality_core) {
      txtContent += `PERSONALITY CORE:\n${detailed.personality_core}\n\n`;
    }
    
    if (detailed.thought_patterns) {
      txtContent += `THOUGHT PATTERNS:\n${detailed.thought_patterns}\n\n`;
    }
    
    if (detailed.emotional_intelligence) {
      txtContent += `EMOTIONAL INTELLIGENCE:\n${detailed.emotional_intelligence}\n\n`;
    }
    
    if (detailed.speech_analysis) {
      txtContent += 'SPEECH ANALYSIS:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.speech_analysis.key_quotes && detailed.speech_analysis.key_quotes.length > 0) {
        txtContent += 'Key Quotes:\n';
        detailed.speech_analysis.key_quotes.forEach((quote: string, i: number) => {
          txtContent += `${i + 1}. "${quote}"\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.speech_analysis.vocabulary_analysis) {
        txtContent += `Vocabulary Analysis: ${detailed.speech_analysis.vocabulary_analysis}\n\n`;
      }
      
      if (detailed.speech_analysis.speech_patterns) {
        txtContent += `Speech Patterns: ${detailed.speech_analysis.speech_patterns}\n\n`;
      }
      
      if (detailed.speech_analysis.emotional_tone) {
        txtContent += `Emotional Tone: ${detailed.speech_analysis.emotional_tone}\n\n`;
      }
    }
    
    if (detailed.visual_evidence) {
      txtContent += 'VISUAL EVIDENCE:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.visual_evidence.facial_expressions) {
        txtContent += `Facial Expressions: ${detailed.visual_evidence.facial_expressions}\n\n`;
      }
      
      if (detailed.visual_evidence.body_language) {
        txtContent += `Body Language: ${detailed.visual_evidence.body_language}\n\n`;
      }
      
      if (detailed.visual_evidence.emotional_indicators) {
        txtContent += `Emotional Indicators: ${detailed.visual_evidence.emotional_indicators}\n\n`;
      }
    }
    
    if (detailed.professional_insights) {
      txtContent += `PROFESSIONAL INSIGHTS:\n${detailed.professional_insights}\n\n`;
    }
    
    if (detailed.relationships) {
      txtContent += 'RELATIONSHIPS:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += `Current Status: ${detailed.relationships.current_status || 'Not specified'}\n`;
      txtContent += `Parental Status: ${detailed.relationships.parental_status || 'Not specified'}\n`;
      txtContent += `Ideal Partner: ${detailed.relationships.ideal_partner || 'Not specified'}\n\n`;
    }
    
    if (detailed.growth_areas) {
      txtContent += 'GROWTH AREAS:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.growth_areas.strengths) {
        txtContent += 'Strengths:\n';
        detailed.growth_areas.strengths.forEach((strength: string) => {
          txtContent += `• ${strength}\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.growth_areas.challenges) {
        txtContent += 'Challenges:\n';
        detailed.growth_areas.challenges.forEach((challenge: string) => {
          txtContent += `• ${challenge}\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.growth_areas.development_path) {
        txtContent += `Development Path: ${detailed.growth_areas.development_path}\n\n`;
      }
    }
  }
  
  // Add comprehensive analysis if available (40-parameter system)
  if (personalityInsights.comprehensiveAnalysis) {
    txtContent += '40-PARAMETER COMPREHENSIVE ANALYSIS:\n';
    txtContent += '='.repeat(60) + '\n\n';
    
    // Cognitive Parameters (20)
    if (personalityInsights.comprehensiveAnalysis.cognitiveAnalysis) {
      txtContent += 'COGNITIVE PARAMETERS (20):\n';
      txtContent += '-'.repeat(40) + '\n';
      
      Object.entries(personalityInsights.comprehensiveAnalysis.cognitiveAnalysis).forEach(([paramId, paramData]: [string, any]) => {
        if (paramData && personalityInsights.cognitiveParameters) {
          const paramInfo = personalityInsights.cognitiveParameters.find((p: any) => p.id === paramId);
          if (paramInfo) {
            txtContent += `\n${paramInfo.name.toUpperCase()}:\n`;
            txtContent += `Score: ${paramData.score || 'N/A'}/100\n`;
            txtContent += `Analysis: ${paramData.analysis || 'No analysis available'}\n`;
            
            if (paramData.quotations && paramData.quotations.length > 0) {
              txtContent += 'Key Quotations:\n';
              paramData.quotations.forEach((quote: string, i: number) => {
                txtContent += `  ${i + 1}. "${quote}"\n`;
              });
            }
            
            if (paramData.evidence) {
              txtContent += `Evidence: ${paramData.evidence}\n`;
            }
            txtContent += '\n';
          }
        }
      });
    }
    
    // Psychological Parameters (20)
    if (personalityInsights.comprehensiveAnalysis.psychologicalAnalysis) {
      txtContent += '\nPSYCHOLOGICAL PARAMETERS (20):\n';
      txtContent += '-'.repeat(40) + '\n';
      
      Object.entries(personalityInsights.comprehensiveAnalysis.psychologicalAnalysis).forEach(([paramId, paramData]: [string, any]) => {
        if (paramData && personalityInsights.psychologicalParameters) {
          const paramInfo = personalityInsights.psychologicalParameters.find((p: any) => p.id === paramId);
          if (paramInfo) {
            txtContent += `\n${paramInfo.name.toUpperCase()}:\n`;
            txtContent += `Score: ${paramData.score || 'N/A'}/100\n`;
            txtContent += `Analysis: ${paramData.analysis || 'No analysis available'}\n`;
            
            if (paramData.quotations && paramData.quotations.length > 0) {
              txtContent += 'Key Quotations:\n';
              paramData.quotations.forEach((quote: string, i: number) => {
                txtContent += `  ${i + 1}. "${quote}"\n`;
              });
            }
            
            if (paramData.evidence) {
              txtContent += `Evidence: ${paramData.evidence}\n`;
            }
            txtContent += '\n';
          }
        }
      });
    }
    
    // Overall Summary
    if (personalityInsights.comprehensiveAnalysis.overallSummary) {
      txtContent += '\nOVERALL SUMMARY:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += personalityInsights.comprehensiveAnalysis.overallSummary + '\n\n';
    }
  }
  
  // Add analysis messages if available
  if ((analysis as any).messages && Array.isArray((analysis as any).messages) && (analysis as any).messages.length > 0) {
    txtContent += 'ANALYSIS MESSAGES:\n';
    txtContent += '='.repeat(60) + '\n\n';
    
    (analysis as any).messages.forEach((message: any, index: number) => {
      if (message.role === 'assistant' && message.content) {
        txtContent += `Message ${index + 1}:\n`;
        txtContent += '-'.repeat(20) + '\n';
        // Clean up message content by removing markdown formatting
        const cleanContent = message.content
          .replace(/#{1,6}\s*/g, '') // Remove headers
          .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
          .replace(/\*(.*?)\*/g, '$1') // Remove italic
          .replace(/`(.*?)`/g, '$1') // Remove code blocks
          .replace(/^\s*[-*+]\s+/gm, '• ') // Convert bullets to simple format
          .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered lists
          .trim();
        txtContent += cleanContent + '\n\n';
      }
    });
  }
  
  txtContent += '='.repeat(80) + '\n';
  txtContent += 'END OF REPORT\n';
  txtContent += '='.repeat(80) + '\n';
  
  return txtContent;
}

// Function to generate HTML for PDF
export function generateAnalysisHtml(analysis: Analysis): string {
  // Extract the personality insights
  const personalityInsights = analysis.personalityInsights as any || {};
  
  // Determine if we have a multi-person analysis
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  // Get total people count
  const peopleCount = personalityInsights.peopleCount || 1;
  
  // For backward compatibility with single-person analysis
  let summary = 'No summary available';
  let detailedAnalysis = {} as any;

  // Format content differently based on single vs. multiple people
  let htmlContent = `
    <html>
    <head>
      <style>
        body {
          font-family: Arial, sans-serif;
          line-height: 1.6;
          color: #333;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        h1 {
          color: #2563eb;
          border-bottom: 1px solid #ddd;
          padding-bottom: 10px;
        }
        h2 {
          color: #4338ca;
          margin-top: 25px;
        }
        h3 {
          color: #1e40af;
        }
        .summary {
          background-color: #f9fafb;
          border-left: 4px solid #3b82f6;
          padding: 15px;
          margin: 20px 0;
        }
        .video-analysis {
          background-color: #fff7ed;
          border-left: 4px solid #f59e0b;
          padding: 20px;
          margin: 20px 0;
          border-radius: 4px;
        }
        .profile {
          background-color: #f3f4f6;
          border-radius: 8px;
          padding: 15px;
          margin: 20px 0;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section {
          margin: 15px 0;
        }
        .footer {
          margin-top: 40px;
          font-size: 0.8em;
          color: #6b7280;
          text-align: center;
        }
      </style>
    </head>
    <body>
      <h1>${analysis.title || 'Personality Analysis Report'}</h1>
      <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
      <p><strong>Analysis Type:</strong> ${analysis.mediaType.charAt(0).toUpperCase() + analysis.mediaType.slice(1)} Analysis</p>
      <p><strong>People Detected:</strong> ${peopleCount}</p>
  `;

  // Check for video analysis data and include it
  if (personalityInsights.videoAnalysis && personalityInsights.videoAnalysis.analysisText) {
    // Clean up analysis text by removing markdown formatting
    const cleanAnalysisText = personalityInsights.videoAnalysis.analysisText
      .replace(/#{1,6}\s*/g, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove code blocks
      .replace(/^\s*[-*+]\s+/gm, '• ') // Convert bullets to simple format
      .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered lists
      .trim();

    htmlContent += `
      <div class="video-analysis">
        <h2>Comprehensive Psychoanalytic Assessment</h2>
        <p><strong>Analyzed Segment:</strong> ${personalityInsights.videoAnalysis.segmentInfo?.label || 'N/A'}</p>
        <p><strong>Duration:</strong> ${personalityInsights.videoAnalysis.segmentInfo?.duration || 'N/A'} seconds</p>
        <p><strong>AI Model:</strong> ${personalityInsights.videoAnalysis.model || 'N/A'}</p>
        
        <div class="section">
          <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">${cleanAnalysisText}</pre>
        </div>
        
        ${personalityInsights.videoAnalysis.audioTranscription?.transcription ? `
        <div class="section">
          <h3>Audio Transcription</h3>
          <p><em>"${personalityInsights.videoAnalysis.audioTranscription.transcription}"</em></p>
        </div>
        ` : ''}
      </div>
    `;
  }
  // Check for comprehensive 40-parameter analysis
  if (personalityInsights.comprehensiveAnalysis) {
    htmlContent += `
      <div class="video-analysis">
        <h2>Comprehensive 40-Parameter Analysis</h2>
        
        <div class="section">
          <h3>Cognitive Parameters (20)</h3>
    `;
    
    // Add cognitive parameters
    if (personalityInsights.comprehensiveAnalysis.cognitiveAnalysis && personalityInsights.cognitiveParameters) {
      Object.entries(personalityInsights.comprehensiveAnalysis.cognitiveAnalysis).forEach(([paramId, paramData]: [string, any]) => {
        const paramInfo = personalityInsights.cognitiveParameters.find((p: any) => p.id === paramId);
        if (paramInfo && paramData) {
          htmlContent += `
            <div class="profile">
              <h4>${paramInfo.name}</h4>
              <p><strong>Score:</strong> ${paramData.score || 'N/A'}/100</p>
              <p><strong>Analysis:</strong> ${paramData.analysis || 'No analysis available'}</p>
              ${paramData.quotations && paramData.quotations.length > 0 ? `
                <p><strong>Key Quotations:</strong></p>
                <ul>
                  ${paramData.quotations.map((quote: string) => `<li><em>"${quote}"</em></li>`).join('')}
                </ul>
              ` : ''}
              ${paramData.evidence ? `<p><strong>Evidence:</strong> ${paramData.evidence}</p>` : ''}
            </div>
          `;
        }
      });
    }
    
    htmlContent += `
        </div>
        
        <div class="section">
          <h3>Psychological Parameters (20)</h3>
    `;
    
    // Add psychological parameters
    if (personalityInsights.comprehensiveAnalysis.psychologicalAnalysis && personalityInsights.psychologicalParameters) {
      Object.entries(personalityInsights.comprehensiveAnalysis.psychologicalAnalysis).forEach(([paramId, paramData]: [string, any]) => {
        const paramInfo = personalityInsights.psychologicalParameters.find((p: any) => p.id === paramId);
        if (paramInfo && paramData) {
          htmlContent += `
            <div class="profile">
              <h4>${paramInfo.name}</h4>
              <p><strong>Score:</strong> ${paramData.score || 'N/A'}/100</p>
              <p><strong>Analysis:</strong> ${paramData.analysis || 'No analysis available'}</p>
              ${paramData.quotations && paramData.quotations.length > 0 ? `
                <p><strong>Key Quotations:</strong></p>
                <ul>
                  ${paramData.quotations.map((quote: string) => `<li><em>"${quote}"</em></li>`).join('')}
                </ul>
              ` : ''}
              ${paramData.evidence ? `<p><strong>Evidence:</strong> ${paramData.evidence}</p>` : ''}
            </div>
          `;
        }
      });
    }
    
    // Add overall summary
    if (personalityInsights.comprehensiveAnalysis.overallSummary) {
      htmlContent += `
        </div>
        
        <div class="section">
          <h3>Overall Summary</h3>
          <div class="summary">
            <p>${personalityInsights.comprehensiveAnalysis.overallSummary}</p>
          </div>
        </div>
      </div>
      `;
    } else {
      htmlContent += '</div></div>';
    }
  }
  // Check for image analysis data (legacy format)
  else if (personalityInsights.comprehensiveAnalysis && typeof personalityInsights.comprehensiveAnalysis === 'string') {
    // Clean up analysis text by removing markdown formatting
    const analysisText = typeof personalityInsights.comprehensiveAnalysis === 'string' 
      ? personalityInsights.comprehensiveAnalysis 
      : JSON.stringify(personalityInsights.comprehensiveAnalysis);
    const cleanAnalysisText = analysisText
      .replace(/#{1,6}\s*/g, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove code blocks
      .replace(/^\s*[-*+]\s+/gm, '• ') // Convert bullets to simple format
      .replace(/^\s*\d+\.\s+/gm, '') // Remove numbered lists
      .trim();

    htmlContent += `
      <div class="video-analysis">
        <h2>Comprehensive Psychoanalytic Assessment</h2>
        <p><strong>Media Type:</strong> Image Analysis</p>
        <p><strong>AI Model:</strong> ${personalityInsights.model || 'N/A'}</p>
        
        <div class="section">
          <pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">${cleanAnalysisText}</pre>
        </div>
      </div>
    `;
  }
  
  if (isMultiPersonAnalysis) {
    // Handle multi-person analysis
    const profiles = personalityInsights.individualProfiles || [];
    const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
    
    htmlContent += `
      <div class="summary">
        <h2>Overview (${peopleCount} People Detected)</h2>
        <p>${overviewSummary}</p>
      </div>
    `;
    
    if (personalityInsights.groupDynamics) {
      htmlContent += `
        <div class="section">
          <h2>Group Dynamics</h2>
          <p>${personalityInsights.groupDynamics}</p>
        </div>
      `;
    }
    
    // Add individual profiles section
    htmlContent += `<h2>Individual Profiles</h2>`;
    
    // Add each person's profile
    profiles.forEach((profile: any, index: number) => {
      const personLabel = profile.personLabel || `Person ${index + 1}`;
      const personSummary = profile.summary || 'No summary available';
      const personDetails = profile.detailed_analysis || {};
      
      htmlContent += `
        <div class="profile">
          <h3>${personLabel}</h3>
          
          <div class="section">
            <h4>Summary</h4>
            <p>${personSummary}</p>
          </div>
          
          <div class="section">
            <h4>Core Personality</h4>
            <p>${personDetails.personality_core || 'Not available'}</p>
          </div>
          
          <div class="section">
            <h4>Thought Patterns</h4>
            <p>${personDetails.thought_patterns || 'Not available'}</p>
          </div>
          
          <div class="section">
            <h4>Professional Insights</h4>
            <p>${personDetails.professional_insights || 'Not available'}</p>
          </div>
      `;
      
      if (personDetails.growth_areas) {
        htmlContent += `
          <div class="section">
            <h4>Key Strengths</h4>
            <ul>
              ${Array.isArray(personDetails.growth_areas.strengths) 
                ? personDetails.growth_areas.strengths.map((s: string) => `<li>${s}</li>`).join('') 
                : '<li>Not available</li>'}
            </ul>
            
            <h4>Challenges</h4>
            <ul>
              ${Array.isArray(personDetails.growth_areas.challenges) 
                ? personDetails.growth_areas.challenges.map((c: string) => `<li>${c}</li>`).join('') 
                : '<li>Not available</li>'}
            </ul>
          </div>
        `;
      }
      
      htmlContent += `</div>`;
    });
    
  } else {
    // Get data for single-person format
    if (personalityInsights.individualProfiles?.length === 1) {
      const profile = personalityInsights.individualProfiles[0];
      summary = profile.summary || 'No summary available';
      detailedAnalysis = profile.detailed_analysis || {};
    } else {
      // Legacy format
      summary = personalityInsights.summary || 'No summary available';
      detailedAnalysis = personalityInsights.detailed_analysis || {};
    }
    
    // Generate single-person content
    htmlContent += `
      <div class="summary">
        <h2>Summary</h2>
        <p>${summary}</p>
      </div>

      <div class="section">
        <h2>Core Personality Traits</h2>
        <p>${detailedAnalysis.personality_core || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Thought Patterns</h2>
        <p>${detailedAnalysis.thought_patterns || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Cognitive Style</h2>
        <p>${detailedAnalysis.cognitive_style || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Professional Insights</h2>
        <p>${detailedAnalysis.professional_insights || 'Not available'}</p>
      </div>
    `;
    
    if (detailedAnalysis.growth_areas) {
      htmlContent += `
        <div class="section">
          <h2>Growth Areas</h2>
          
          <h3>Strengths</h3>
          <ul>
            ${(detailedAnalysis.growth_areas.strengths || []).map((s: string) => `<li>${s}</li>`).join('')}
          </ul>
          
          <h3>Challenges</h3>
          <ul>
            ${(detailedAnalysis.growth_areas.challenges || []).map((c: string) => `<li>${c}</li>`).join('')}
          </ul>
          
          <h3>Development Path</h3>
          <p>${detailedAnalysis.growth_areas.development_path || 'Not available'}</p>
        </div>
      `;
    }
  }
  
  htmlContent += `
      <div class="footer">
        <p>Generated by Personality Insights App</p>
        <p>${new Date().toISOString()}</p>
      </div>
    </body>
    </html>
  `;
  
  return htmlContent;
}

// Generate DOCX document for an analysis
export async function generateDocx(analysis: Analysis): Promise<Buffer> {
  // Extract the personality insights
  const personalityInsights = analysis.personalityInsights as any || {};
  
  // Determine if we have a multi-person analysis
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  // Get total people count
  const peopleCount = personalityInsights.peopleCount || 1;
  
  // Document content
  const doc = new Document({
    title: analysis.title || 'Personality Analysis Report',
    description: 'AI-generated personality analysis',
    sections: [{
      properties: {},
      children: []
    }],
  });

  const children = [];

  // Title
  children.push(
    new Paragraph({
      text: analysis.title || 'Personality Analysis Report',
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
    })
  );

  // Date and Type
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'Date: ', bold: true }),
        new TextRun(new Date().toLocaleDateString()),
      ],
    })
  );

  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'Analysis Type: ', bold: true }),
        new TextRun(analysis.mediaType.charAt(0).toUpperCase() + analysis.mediaType.slice(1)),
      ],
    })
  );

  children.push(new Paragraph({})); // Spacing

  if (isMultiPersonAnalysis) {
    // Handle multi-person analysis
    const profiles = personalityInsights.individualProfiles || [];
    const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
    
    // Overview
    children.push(
      new Paragraph({
        text: `Overview (${peopleCount} People Detected)`,
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: overviewSummary }));
    
    // Group Dynamics if available
    if (personalityInsights.groupDynamics) {
      children.push(
        new Paragraph({
          text: 'Group Dynamics',
          heading: HeadingLevel.HEADING_2,
        })
      );
      
      children.push(new Paragraph({ text: personalityInsights.groupDynamics }));
    }
    
    // Individual Profiles
    children.push(
      new Paragraph({
        text: 'Individual Profiles',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    // Add each person's profile
    profiles.forEach((profile: any, index: number) => {
      const personLabel = profile.personLabel || `Person ${index + 1}`;
      const personSummary = profile.summary || 'No summary available';
      const personDetails = profile.detailed_analysis || {};
      
      children.push(
        new Paragraph({
          text: personLabel,
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      // Summary
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Summary', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personSummary }));
      
      // Core Personality
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Core Personality', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.personality_core || 'Not available' }));
      
      // Thought Patterns
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Thought Patterns', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.thought_patterns || 'Not available' }));
      
      // Professional Insights
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Professional Insights', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.professional_insights || 'Not available' }));
      
      // Add comprehensive analysis if available
      if (personalityInsights.comprehensiveAnalysis && personalityInsights.comprehensiveAnalysis.cognitiveAnalysis) {
        children.push(
          new Paragraph({
            text: 'Comprehensive 40-Parameter Analysis',
            heading: HeadingLevel.HEADING_2,
          })
        );
        
        // Cognitive Parameters
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Cognitive Parameters (20)', bold: true })],
          })
        );
        
        if (personalityInsights.cognitiveParameters) {
          Object.entries(personalityInsights.comprehensiveAnalysis.cognitiveAnalysis).forEach(([paramId, paramData]: [string, any]) => {
            const paramInfo = personalityInsights.cognitiveParameters.find((p: any) => p.id === paramId);
            if (paramInfo && paramData) {
              children.push(
                new Paragraph({
                  children: [new TextRun({ text: paramInfo.name, bold: true })],
                })
              );
              
              children.push(new Paragraph({ text: `Score: ${paramData.score || 'N/A'}/100` }));
              children.push(new Paragraph({ text: `Analysis: ${paramData.analysis || 'No analysis available'}` }));
              
              if (paramData.quotations && paramData.quotations.length > 0) {
                children.push(new Paragraph({ text: 'Key Quotations:' }));
                paramData.quotations.forEach((quote: string) => {
                  children.push(new Paragraph({ text: `• "${quote}"` }));
                });
              }
              
              if (paramData.evidence) {
                children.push(new Paragraph({ text: `Evidence: ${paramData.evidence}` }));
              }
              
              children.push(new Paragraph({})); // Spacing
            }
          });
        }
        
        // Psychological Parameters
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Psychological Parameters (20)', bold: true })],
          })
        );
        
        if (personalityInsights.psychologicalParameters) {
          Object.entries(personalityInsights.comprehensiveAnalysis.psychologicalAnalysis || {}).forEach(([paramId, paramData]: [string, any]) => {
            const paramInfo = personalityInsights.psychologicalParameters.find((p: any) => p.id === paramId);
            if (paramInfo && paramData) {
              children.push(
                new Paragraph({
                  children: [new TextRun({ text: paramInfo.name, bold: true })],
                })
              );
              
              children.push(new Paragraph({ text: `Score: ${paramData.score || 'N/A'}/100` }));
              children.push(new Paragraph({ text: `Analysis: ${paramData.analysis || 'No analysis available'}` }));
              
              if (paramData.quotations && paramData.quotations.length > 0) {
                children.push(new Paragraph({ text: 'Key Quotations:' }));
                paramData.quotations.forEach((quote: string) => {
                  children.push(new Paragraph({ text: `• "${quote}"` }));
                });
              }
              
              if (paramData.evidence) {
                children.push(new Paragraph({ text: `Evidence: ${paramData.evidence}` }));
              }
              
              children.push(new Paragraph({})); // Spacing
            }
          });
        }
        
        // Overall Summary
        if (personalityInsights.comprehensiveAnalysis.overallSummary) {
          children.push(
            new Paragraph({
              children: [new TextRun({ text: 'Overall Summary', bold: true })],
            })
          );
          children.push(new Paragraph({ text: personalityInsights.comprehensiveAnalysis.overallSummary }));
        }
      }
      
      // Strengths and Challenges if available
      if (personDetails.growth_areas) {
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Key Strengths', bold: true })],
          })
        );
        
        if (Array.isArray(personDetails.growth_areas.strengths)) {
          personDetails.growth_areas.strengths.forEach((strength: string) => {
            children.push(new Paragraph({ text: `• ${strength}` }));
          });
        } else {
          children.push(new Paragraph({ text: '• Not available' }));
        }
        
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Challenges', bold: true })],
          })
        );
        
        if (Array.isArray(personDetails.growth_areas.challenges)) {
          personDetails.growth_areas.challenges.forEach((challenge: string) => {
            children.push(new Paragraph({ text: `• ${challenge}` }));
          });
        } else {
          children.push(new Paragraph({ text: '• Not available' }));
        }
      }
      
      // Add spacing after each profile
      children.push(new Paragraph({}));
    });
    
  } else {
    // Get data for single-person format
    let summary = 'No summary available';
    let detailedAnalysis = {} as any;
    
    if (personalityInsights.individualProfiles?.length === 1) {
      const profile = personalityInsights.individualProfiles[0];
      summary = profile.summary || 'No summary available';
      detailedAnalysis = profile.detailed_analysis || {};
    } else {
      // Legacy format
      summary = personalityInsights.summary || 'No summary available';
      detailedAnalysis = personalityInsights.detailed_analysis || {};
    }
    
    // Summary
    children.push(
      new Paragraph({
        text: 'Summary',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: summary }));
    
    // Core Personality
    children.push(
      new Paragraph({
        text: 'Core Personality Traits',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.personality_core || 'Not available' }));
    
    // Thought Patterns
    children.push(
      new Paragraph({
        text: 'Thought Patterns',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.thought_patterns || 'Not available' }));
    
    // Cognitive Style
    children.push(
      new Paragraph({
        text: 'Cognitive Style',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.cognitive_style || 'Not available' }));
    
    // Professional Insights
    children.push(
      new Paragraph({
        text: 'Professional Insights',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.professional_insights || 'Not available' }));
    
    // Growth Areas if available
    if (detailedAnalysis.growth_areas) {
      children.push(
        new Paragraph({
          text: 'Growth Areas',
          heading: HeadingLevel.HEADING_2,
        })
      );
      
      // Strengths
      children.push(
        new Paragraph({
          text: 'Strengths',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      if (Array.isArray(detailedAnalysis.growth_areas.strengths)) {
        detailedAnalysis.growth_areas.strengths.forEach((strength: string) => {
          children.push(new Paragraph({ text: `• ${strength}` }));
        });
      } else {
        children.push(new Paragraph({ text: '• Not available' }));
      }
      
      // Challenges
      children.push(
        new Paragraph({
          text: 'Challenges',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      if (Array.isArray(detailedAnalysis.growth_areas.challenges)) {
        detailedAnalysis.growth_areas.challenges.forEach((challenge: string) => {
          children.push(new Paragraph({ text: `• ${challenge}` }));
        });
      } else {
        children.push(new Paragraph({ text: '• Not available' }));
      }
      
      // Development Path
      children.push(
        new Paragraph({
          text: 'Development Path',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      children.push(new Paragraph({ text: detailedAnalysis.growth_areas.development_path || 'Not available' }));
    }
  }
  
  // Footer
  children.push(new Paragraph({})); // Spacing
  children.push(
    new Paragraph({
      text: 'Generated by Personality Insights App',
      alignment: AlignmentType.CENTER,
    })
  );
  
  children.push(
    new Paragraph({
      text: new Date().toISOString(),
      alignment: AlignmentType.CENTER,
    })
  );
  
  // Create a new document with proper structure
  const newDoc = new Document({
    sections: [{
      properties: {},
      children: children
    }]
  });
  
  return await Packer.toBuffer(newDoc);
}

// Create PDF from HTML content
export async function generatePdf(htmlContent: string): Promise<Buffer> {
  let browser;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    await page.setContent(htmlContent, { waitUntil: 'networkidle0' });
    
    const pdfBuffer = await page.pdf({
      format: 'A4',
      margin: {
        top: '0.5in',
        right: '0.5in',
        bottom: '0.5in',
        left: '0.5in'
      },
      printBackground: true
    });
    
    return Buffer.from(pdfBuffer);
  } catch (error) {
    throw new Error(`PDF generation failed: ${error}`);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}