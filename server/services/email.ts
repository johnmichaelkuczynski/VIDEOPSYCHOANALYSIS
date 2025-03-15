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
}

export async function sendAnalysisEmail({
  share,
  analysis,
}: SendAnalysisEmailParams): Promise<boolean> {
  try {
    // If SendGrid is not configured, log error and return false
    if (!process.env.SENDGRID_API_KEY) {
      console.error('SendGrid is not configured. Emails cannot be sent.');
      return false;
    }

    const { personalityInsights } = analysis;
    const summary = personalityInsights?.summary || 'No summary available';
    const detailedAnalysis = personalityInsights?.detailed_analysis || {};

    const emailContent = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #333;">Your Personality Analysis Results</h2>

        <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
          <h3 style="color: #444;">Summary</h3>
          <p>${summary}</p>
        </div>

        <div style="margin: 20px 0;">
          <h3 style="color: #444;">Core Personality Traits</h3>
          <p>${detailedAnalysis.personality_core || 'Not available'}</p>
        </div>

        <div style="margin: 20px 0;">
          <h3 style="color: #444;">Professional Insights</h3>
          <p>${detailedAnalysis.professional_insights || 'Not available'}</p>
        </div>

        <div style="margin: 20px 0;">
          <h3 style="color: #444;">Growth Areas</h3>
          <ul>
            ${(detailedAnalysis.growth_areas?.strengths || []).map((s: string) => `<li>${s}</li>`).join('')}
          </ul>
        </div>

        <div style="background: #eef4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
          <p style="font-style: italic;">This analysis was shared by: ${share.senderEmail}</p>
        </div>
      </div>
    `;

    await mailService.send({
      to: share.recipientEmail,
      from: {
        email: FROM_EMAIL,
        name: "Personality Insights"
      },
      subject: 'Your Personality Analysis Results',
      html: emailContent,
    });

    return true;
  } catch (error) {
    console.error('SendGrid email error:', error);
    const sendGridError = error as any;
    if (sendGridError.response?.body?.errors) {
      console.error('SendGrid detailed errors:', sendGridError.response.body.errors);
    }
    return false;
  }
}