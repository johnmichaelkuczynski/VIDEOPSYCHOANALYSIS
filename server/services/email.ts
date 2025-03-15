import { MailService } from '@sendgrid/mail';
import { Share, Analysis } from '../../shared/schema';

if (!process.env.SENDGRID_API_KEY) {
  throw new Error("SENDGRID_API_KEY environment variable must be set");
}

const mailService = new MailService();
mailService.setApiKey(process.env.SENDGRID_API_KEY);

const FROM_EMAIL = 'no-reply@personality-insights.app'; // Update this with your verified sender

interface SendAnalysisEmailParams {
  share: Share;
  analysis: Analysis;
}

export async function sendAnalysisEmail({
  share,
  analysis,
}: SendAnalysisEmailParams): Promise<boolean> {
  try {
    const personalityTraits = analysis.personalityInsights;
    const faceAttributes = analysis.faceAnalysis;

    const emailContent = `
      <h2>Personality Insights Analysis Results</h2>
      <p>Someone has shared a personality analysis with you!</p>
      
      <h3>Face Analysis Results:</h3>
      <pre>${JSON.stringify(faceAttributes, null, 2)}</pre>
      
      <h3>Personality Insights:</h3>
      <pre>${JSON.stringify(personalityTraits, null, 2)}</pre>
      
      <p>This analysis was shared by: ${share.senderEmail}</p>
    `;

    await mailService.send({
      to: share.recipientEmail,
      from: FROM_EMAIL,
      subject: 'Your Personality Insights Analysis Results',
      html: emailContent,
    });

    return true;
  } catch (error) {
    console.error('SendGrid email error:', error);
    return false;
  }
}
