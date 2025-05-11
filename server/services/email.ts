import { MailService } from '@sendgrid/mail';
import { Share, Analysis } from '../../shared/schema';

// Check if required environment variables are set
if (!process.env.SENDGRID_API_KEY) {
  console.warn("Warning: SENDGRID_API_KEY environment variable is not set. Email functionality will be disabled.");
}

if (!process.env.SENDGRID_VERIFIED_SENDER) {
  console.warn("Warning: SENDGRID_VERIFIED_SENDER environment variable is not set. Using default sender.");
}

const mailService = new MailService();
if (process.env.SENDGRID_API_KEY) {
  mailService.setApiKey(process.env.SENDGRID_API_KEY);
}

const FROM_EMAIL = process.env.SENDGRID_VERIFIED_SENDER || 'notifications@personality-insights.app';

interface SendAnalysisEmailParams {
  share: Share;
  analysis: Analysis;
  shareUrl?: string;
}

export async function sendAnalysisEmail({
  share,
  analysis,
  shareUrl
}: SendAnalysisEmailParams): Promise<boolean> {
  try {
    // If SendGrid is not configured, log error and return false
    if (!process.env.SENDGRID_API_KEY) {
      console.error('SendGrid is not configured. Emails cannot be sent.');
      return false;
    }
    
    console.log(`[SendGrid] Attempting to send email from ${FROM_EMAIL} to ${share.recipientEmail}`);
    console.log(`[SendGrid] Using analysis ID: ${analysis.id}, share ID: ${share.id}`);
    console.log(`[SendGrid] Share URL: ${shareUrl}`);
    

    // Parse the JSON data from the database
    const personalityInsights = analysis.personalityInsights as any || {};
    
    // Detect if we have a multi-person analysis
    const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                                  Array.isArray(personalityInsights.individualProfiles) && 
                                  personalityInsights.individualProfiles.length > 1;
    
    // Get total people count
    const peopleCount = personalityInsights.peopleCount || 1;
    
    // For backward compatibility with single-person analysis
    let summary = 'No summary available';
    let detailedAnalysis = {};

    // Determine if this was a video analysis that includes transcription
    const isVideoAnalysis = analysis.mediaType === 'video';
    
    // Get transcription data - it could be stored in multiple places based on our implementation
    // First, check if it's stored directly in audioTranscription field
    let transcription = '';
    if (analysis.audioTranscription) {
      const audioData = analysis.audioTranscription as any;
      transcription = audioData.transcription || '';
    } 
    // If not found, check if it's nested in videoAnalysis
    else if (analysis.videoAnalysis) {
      const videoData = analysis.videoAnalysis as any;
      if (videoData.audioTranscription) {
        transcription = videoData.audioTranscription.transcription || '';
      }
    }
    
    // Format content differently based on single vs. multiple people
    let emailContent = '';
    
    if (isMultiPersonAnalysis) {
      // Handle multi-person analysis
      const profiles = personalityInsights.individualProfiles || [];
      const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
      
      // Generate group analysis HTML
      emailContent = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333;">Your Multi-Person Analysis Results</h2>
          
          <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #444;">Overview (${peopleCount} People Detected)</h3>
            <p>${overviewSummary}</p>
          </div>
          
          ${isVideoAnalysis && transcription ? `
          <div style="background: #f5f5ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #6366f1;">
            <h3 style="color: #4338ca;">Video Speech Transcription</h3>
            <p style="font-style: italic;">"${transcription}"</p>
          </div>
          ` : ''}
          
          ${personalityInsights.groupDynamics ? `
          <div style="background: #f0fff4; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #10b981;">
            <h3 style="color: #059669;">Group Dynamics</h3>
            <p>${personalityInsights.groupDynamics}</p>
          </div>
          ` : ''}
      `;
      
      // Add individual profiles section
      emailContent += `
        <h3 style="color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px;">Individual Profiles</h3>
      `;
      
      // Add each person's profile
      profiles.forEach((profile: any, index: number) => {
        const personLabel = profile.personLabel || `Person ${index + 1}`;
        const personSummary = profile.summary || 'No summary available';
        const personDetails = profile.detailed_analysis || {};
        
        emailContent += `
          <div style="margin: 30px 0; background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <h3 style="color: #3b82f6; margin-top: 0;">${personLabel}</h3>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Summary</h4>
              <p>${personSummary}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Core Personality</h4>
              <p>${personDetails.personality_core || 'Not available'}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Professional Insights</h4>
              <p>${personDetails.professional_insights || 'Not available'}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Key Strengths</h4>
              <ul style="margin-top: 5px;">
                ${Array.isArray(personDetails.growth_areas?.strengths) 
                  ? personDetails.growth_areas.strengths.map((s: string) => `<li>${s}</li>`).join('') 
                  : '<li>Not available</li>'}
              </ul>
            </div>
          </div>
        `;
      });
      
      // Add footer
      emailContent += `
        <div style="background: #eef4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
          <p style="font-style: italic;">This multi-person analysis was shared by: ${share.senderEmail}</p>
          <p style="font-size: 0.8em; color: #666; margin-top: 10px;">Analysis based on ${isVideoAnalysis ? 'video' : 'image'} processing with advanced AI technology.</p>
          ${shareUrl ? `<p style="margin-top: 15px;"><a href="${shareUrl}" style="display: inline-block; background: #3b82f6; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-weight: 500;">View Analysis Online</a></p>` : ''}
        </div>
      </div>
      `;
      
    } else {
      // Get data for single-person format (traditional format)
      if (personalityInsights.individualProfiles?.length === 1) {
        const profile = personalityInsights.individualProfiles[0];
        summary = profile.summary || 'No summary available';
        detailedAnalysis = profile.detailed_analysis || {};
      } else {
        // Legacy format
        summary = personalityInsights.summary || 'No summary available';
        detailedAnalysis = personalityInsights.detailed_analysis || {};
      }
      
      // Generate single-person email content (original format)
      emailContent = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333;">Your Personality Analysis Results</h2>
          
          <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #444;">Summary</h3>
            <p>${summary}</p>
          </div>

          ${isVideoAnalysis && transcription ? `
          <div style="background: #f5f5ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #6366f1;">
            <h3 style="color: #4338ca;">Video Speech Transcription</h3>
            <p style="font-style: italic;">"${transcription}"</p>
          </div>
          ` : ''}

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Core Personality Traits</h3>
            <p>${detailedAnalysis.personality_core || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Thought Patterns</h3>
            <p>${detailedAnalysis.thought_patterns || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Professional Insights</h3>
            <p>${detailedAnalysis.professional_insights || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Growth Areas</h3>
            <h4 style="color: #666;">Strengths</h4>
            <ul>
              ${(detailedAnalysis.growth_areas?.strengths || []).map((s: string) => `<li>${s}</li>`).join('')}
            </ul>
            
            <h4 style="color: #666;">Challenges</h4>
            <ul>
              ${(detailedAnalysis.growth_areas?.challenges || []).map((c: string) => `<li>${c}</li>`).join('')}
            </ul>
            
            <h4 style="color: #666;">Development Path</h4>
            <p>${detailedAnalysis.growth_areas?.development_path || 'Not available'}</p>
          </div>

          <div style="background: #eef4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <p style="font-style: italic;">This analysis was shared by: ${share.senderEmail}</p>
            <p style="font-size: 0.8em; color: #666; margin-top: 10px;">Analysis based on ${isVideoAnalysis ? 'video' : 'image'} processing with AI technology.</p>
            ${shareUrl ? `<p style="margin-top: 15px;"><a href="${shareUrl}" style="display: inline-block; background: #3b82f6; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-weight: 500;">View Analysis Online</a></p>` : ''}
          </div>
        </div>
      `;
    }

    // Prepare email data object
    const emailData = {
      to: share.recipientEmail,
      from: {
        email: FROM_EMAIL,
        name: "Personality Insights"
      },
      subject: 'Your Personality Analysis Results',
      html: emailContent,
    };
    
    console.log(`[SendGrid] Preparing to send email with subject: ${emailData.subject}`);
    console.log(`[SendGrid] From: ${emailData.from.name} <${emailData.from.email}>`);
    console.log(`[SendGrid] To: ${emailData.to}`);
    
    try {
      console.log('[SendGrid] Attempting to send email via SendGrid API...');
      const response = await mailService.send(emailData);
      console.log('[SendGrid] Email sent successfully!', response);
      return true;
    } catch (sendError) {
      console.error('[SendGrid] Error while sending email:', sendError);
      return false;
    }
  } catch (error) {
    console.error('[SendGrid] Email preparation error:', error);
    const sendGridError = error as any;
    if (sendGridError.response?.body?.errors) {
      console.error('[SendGrid] SendGrid detailed errors:', sendGridError.response.body.errors);
    }
    return false;
  }
}