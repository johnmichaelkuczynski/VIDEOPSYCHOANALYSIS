// Comprehensive Protocol Analysis Service
// Implements 6 evaluation protocols: Cognitive, Psychological, Psychopathological (Normal + Comprehensive modes)

export interface ProtocolResult {
  protocol: string;
  mode: 'normal' | 'comprehensive';
  phase: 1 | 2 | 3 | 4;
  questions: string[];
  responses: string[];
  scores: Record<string, number>;
  finalScore: number;
  summary: string;
  category: string;
}

export interface ComprehensiveProtocolResult {
  results: ProtocolResult[];
  overallSummary: string;
  totalTime: number;
}

export type ProtocolType = 
  | 'cognitive' 
  | 'psychological' 
  | 'psychopathological'
  | 'comprehensive-cognitive'
  | 'comprehensive-psychological' 
  | 'comprehensive-psychopathological';

// Protocol Questions
const COGNITIVE_QUESTIONS = [
  "Is it insightful?",
  "Does it develop points? (Or, if it is a short excerpt, is there evidence that it would develop points if extended)?",
  "Is the organization merely sequential (just one point after another, little or no logical scaffolding)? Or are the ideas arranged, not just sequentially but hierarchically?",
  "If the points it makes are not insightful, does it operate skillfully with canons of logic/reasoning?",
  "Are the points cliches? Or are they 'fresh'?",
  "Does it use technical jargon to obfuscate or to render more precise?",
  "Is it organic? Do points develop in an organic, natural way? Do they 'unfold'? Or are they forced and artificial?",
  "Does it open up new domains? Or, on the contrary, does it shut off inquiry (by conditionalizing further discussion of the matters on acceptance of its internal and possibly very faulty logic)?",
  "Is it actually intelligent or just the work of somebody who, judging by the subject-matter, is presumed to be intelligent (but may not be)?",
  "Is it real or is it phony?",
  "Do the sentences exhibit complex and coherent internal logic?",
  "Is the passage governed by a strong concept? Or is the only organization driven purely by expository (as opposed to epistemic) norms?",
  "Is there system-level control over ideas? In other words, does the author seem to recall what he said earlier and to be in a position to integrate it into points he has made since then?",
  "Are the points 'real'? Are they fresh? Or is some institution or some accepted vein of propaganda or orthodoxy just using the author as a mouth piece?",
  "Is the writing evasive or direct?",
  "Are the statements ambiguous?",
  "Does the progression of the text develop according to who said what or according to what entails or confirms what?",
  "Does the author use other authors to develop his ideas or to cloak his own lack of ideas?"
];

const PSYCHOLOGICAL_QUESTIONS = [
  "Does the text reveal a stable, coherent self-concept, or is the self fragmented/contradictory?",
  "Is there evidence of ego strength (resilience, capacity to tolerate conflict/ambiguity), or does the psyche rely on brittle defenses?",
  "Are defenses primarily mature (sublimation, humor, anticipation), neurotic (intellectualization, repression), or primitive (splitting, denial, projection)?",
  "Does the writing show integration of affect and thought, or are emotions split off / overly intellectualized?",
  "Is the author's stance defensive/avoidant or direct/engaged?",
  "Does the psyche appear narcissistically organized (grandiosity, fragile self-esteem, hunger for validation), or not?",
  "Are desires/drives expressed openly, displaced, or repressed?",
  "Does the voice suggest internal conflict (superego vs. id, competing identifications), or monolithic certainty?",
  "Is there evidence of object constancy (capacity to sustain nuanced view of others) or splitting (others seen as all-good/all-bad)?",
  "Is aggression integrated (channeled productively) or dissociated/projected?",
  "Is the author capable of irony/self-reflection, or trapped in compulsive earnestness / defensiveness?",
  "Does the text suggest psychological growth potential (openness, curiosity, capacity to metabolize experience) or rigidity?",
  "Is the discourse paranoid / persecutory (others as threats, conspiracies) or reality-based?",
  "Does the tone reflect authentic engagement with reality, or phony simulation of depth?",
  "Is the psyche resilient under stress, or fragile / evasive?",
  "Is there evidence of compulsion or repetition (obsessional returns to the same themes), or flexible progression?",
  "Does the author show capacity for intimacy / genuine connection, or only instrumental/defended relations?",
  "Is shame/guilt worked through constructively or disavowed/projected?"
];

const PSYCHOPATHOLOGICAL_QUESTIONS = [
  "Does the text reveal distorted reality testing (delusion, paranoia, magical thinking), or intact contact with reality?",
  "Is there evidence of persecutory ideation (seeing threats/conspiracies) or is perception proportionate?",
  "Does the subject show rigid obsessional patterns (compulsion, repetitive fixation) vs. flexible thought?",
  "Are there signs of narcissistic pathology (grandiosity, exploitation, lack of empathy), or balanced self-other relation?",
  "Is aggression expressed as sadism, cruelty, destructive glee, or is it integrated/controlled?",
  "Is affect regulation stable or does it suggest lability, rage, despair, manic flight?",
  "Does the person exhibit emptiness, hollowness, anhedonia, or a capacity for meaning/connection?",
  "Is there evidence of identity diffusion (incoherence, role-shifting, lack of stable self)?",
  "Are interpersonal patterns exploitative/manipulative or reciprocal/genuine?",
  "Does the psyche lean toward psychotic organization (loss of boundaries, hallucination-like claims), borderline organization (splitting, fear of abandonment), or neurotic organization (anxiety, repression)?",
  "Are defenses predominantly primitive (denial, projection, splitting) or higher-level?",
  "Is there evidence of pathological lying, phoniness, simulation, or authentic communication?",
  "Does the discourse exhibit compulsive hostility toward norms/authorities (paranoid defiance) or measured critique?",
  "Is sexuality integrated or perverse/displaced (voyeurism, exhibitionism, compulsive control)?",
  "Is the overall presentation coherent and reality-based or chaotic, persecutory, hollow, performative?"
];

import { OpenAI } from 'openai';
import { Anthropic } from '@anthropic-ai/sdk';

// AI Model clients (imported from main routes)
let openai: OpenAI | null = null;
let anthropic: Anthropic | null = null;
let deepseek: OpenAI | null = null;
let perplexity: OpenAI | null = null;

// Initialize AI clients
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

if (process.env.ANTHROPIC_API_KEY) {
  anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}

if (process.env.DEEPSEEK_API_KEY) {
  deepseek = new OpenAI({
    apiKey: process.env.DEEPSEEK_API_KEY,
    baseURL: 'https://api.deepseek.com'
  });
}

if (process.env.PERPLEXITY_API_KEY) {
  perplexity = new OpenAI({
    apiKey: process.env.PERPLEXITY_API_KEY,
    baseURL: 'https://api.perplexity.ai'
  });
}

// AI Model calling function
async function callAIModel(model: string, prompt: string): Promise<string> {
  try {
    if (model === "deepseek" && deepseek) {
      const response = await deepseek.chat.completions.create({
        model: "deepseek-chat",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 4000,
        temperature: 0.7
      });
      return response.choices[0]?.message?.content || "";
    } else if (model === "anthropic" && anthropic) {
      const response = await anthropic.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 4000,
        messages: [{ role: "user", content: prompt }]
      });
      return response.content[0]?.type === 'text' ? response.content[0].text : "";
    } else if (model === "perplexity" && perplexity) {
      const response = await perplexity.chat.completions.create({
        model: "sonar",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 4000,
        temperature: 0.7
      });
      return response.choices[0]?.message?.content || "";
    } else if (openai) {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 4000,
        temperature: 0.7
      });
      return response.choices[0]?.message?.content || "";
    }
  } catch (error) {
    console.error(`AI model ${model} failed:`, error);
    throw new Error(`AI model ${model} failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
  
  throw new Error(`No available AI model for: ${model}`);
}

// Build protocol prompts
function buildPhase1Prompt(text: string, questions: string[], protocolType: string): string {
  const baseInstructions = `
Answer these questions in connection with this text. Also give a score out of 100 for each question.

A score of N/100 (e.g. 73/100) means that (100-N)/100 (e.g. 27/100) outperform the author with respect to the parameter defined by the question.

You are not grading. You are answering questions.

${protocolType === 'cognitive' ? `
You do not use a risk-averse standard; you do not attempt to be diplomatic; you do not attempt to comply with risk-averse, medium-range IQ, academic norms. You do not make assumptions about the level of the paper; it could be a work of the highest excellence and genius, or it could be the work of a moron.

Do not give credit merely for use of jargon or for referencing authorities. Focus on substance. Only give points for scholarly references/jargon if they unambiguously increase substance.

This is not a grading app. Do not penalize boldness. Do not take points away for insights that, if correct, stand on their own.
` : ''}

${protocolType === 'psychological' ? `
Do not default to diagnostic checklists; describe configuration of psyche.

Do not conflate verbal confidence with psychological strength.

Do not penalize honesty, boldness, or extreme statements if they indicate integration rather than breakdown.

Evaluate relative to the general population, not only "advanced" or "pathological" groups.
` : ''}

${protocolType === 'psychopathological' ? `
You are not diagnosing. You are describing the degree of psychopathology revealed.

Do not give credit for rhetorical surface (confidence, erudition). Focus on reality testing, defenses, affect, and interpersonal stance.

Evaluate relative to the general population, not just clinical populations.

Do not penalize intense but integrated thought — pathology is disorganization, not extremity.
` : ''}

Summarize the text and categorize the presentation.

Text: ${text}

Questions:
${questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}`;

  return baseInstructions;
}

function buildPhase2Prompt(text: string, questions: string[], scores: Record<string, number>, protocolType: string): string {
  const lowScores = Object.entries(scores).filter(([_, score]) => score < 95);
  
  if (lowScores.length === 0) {
    return "All scores are 95/100 or higher. Phase 2 pushback is not required.";
  }

  const pushbackText = lowScores.map(([question, score]) => 
    `Your position is that ${100 - score}/100 outperform the author with respect to the ${protocolType} metric defined by the question "${question}". That is your position, am I right? And are you sure about that?`
  ).join('\n\n');

  return `
${pushbackText}

What concrete ${protocolType === 'cognitive' ? 'cognitive' : protocolType === 'psychological' ? 'psychological' : 'pathological resilience'} strengths do those outperformers have that this author lacks?

Answer the following questions about the text de novo:
${questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}

Text: ${text}`;
}

function buildPhase3Prompt(scores: Record<string, number>): string {
  const examples = Object.entries(scores).slice(0, 3).map(([question, score]) => {
    const outperformers = 100 - score;
    return `If you scored ${score}/100 for "${question}", that means ${outperformers}/100 Walmart patrons outperform the author. Provide specific examples of work by those patrons that demonstrates this superiority.`;
  }).join('\n\n');

  return `
You stated numerical scores that mean certain percentages of Walmart patrons outperform the author. Are your numerical scores consistent with this interpretation?

${examples}

If you cannot provide specific examples, justify or revise the scores.`;
}

function buildPhase4Prompt(protocolType: string): string {
  return `
Before finalizing scores, confirm:

${protocolType === 'cognitive' ? `
- Have you penalized the text for not being conventional? If yes, recalibrate.
- Does the score reflect truth density, not compliance with norms?
` : ''}

${protocolType === 'psychological' ? `
- Was the subject penalized for unconventional affect or honesty? If yes → recalibrate.
- Does the score reflect ego integration and authenticity, not mere social compliance?
` : ''}

${protocolType === 'psychopathological' ? `
- Was the subject penalized for boldness or eccentricity rather than pathology? If yes → recalibrate.
- Does the score reflect actual disorganization / dysfunction, not social nonconformity?
` : ''}

- Is the Walmart metric empirically grounded or a lazy guess?

Provide your final analysis and scores.`;
}

// Main protocol execution functions
export async function executeNormalProtocol(
  text: string, 
  protocolType: 'cognitive' | 'psychological' | 'psychopathological',
  model: string = 'deepseek'
): Promise<ProtocolResult> {
  const startTime = Date.now();
  
  let questions: string[];
  switch (protocolType) {
    case 'cognitive':
      questions = COGNITIVE_QUESTIONS;
      break;
    case 'psychological':
      questions = PSYCHOLOGICAL_QUESTIONS;
      break;
    case 'psychopathological':
      questions = PSYCHOPATHOLOGICAL_QUESTIONS;
      break;
  }
  
  const prompt = buildPhase1Prompt(text, questions, protocolType);
  const response = await callAIModel(model, prompt);
  
  // Parse scores from response (simplified - in production would use more robust parsing)
  const scores: Record<string, number> = {};
  questions.forEach((question, i) => {
    // Look for score patterns like "85/100" or "Score: 85"
    const scoreMatch = response.match(new RegExp(`${i + 1}[^0-9]*([0-9]{1,3})/100|Score[^0-9]*([0-9]{1,3})`, 'i'));
    scores[question] = scoreMatch ? parseInt(scoreMatch[1] || scoreMatch[2]) : 75;
  });
  
  const finalScore = Math.round(Object.values(scores).reduce((a, b) => a + b, 0) / Object.values(scores).length);
  
  return {
    protocol: protocolType,
    mode: 'normal',
    phase: 1,
    questions,
    responses: [response],
    scores,
    finalScore,
    summary: response.substring(0, 500) + '...',
    category: extractCategory(response)
  };
}

export async function executeComprehensiveProtocol(
  text: string, 
  protocolType: 'cognitive' | 'psychological' | 'psychopathological',
  model: string = 'deepseek'
): Promise<ProtocolResult> {
  const startTime = Date.now();
  
  let questions: string[];
  switch (protocolType) {
    case 'cognitive':
      questions = COGNITIVE_QUESTIONS;
      break;
    case 'psychological':
      questions = PSYCHOLOGICAL_QUESTIONS;
      break;
    case 'psychopathological':
      questions = PSYCHOPATHOLOGICAL_QUESTIONS;
      break;
  }
  
  const responses: string[] = [];
  let currentScores: Record<string, number> = {};
  
  // Phase 1
  const phase1Prompt = buildPhase1Prompt(text, questions, protocolType);
  const phase1Response = await callAIModel(model, phase1Prompt);
  responses.push(phase1Response);
  
  // Parse initial scores
  questions.forEach((question, i) => {
    const scoreMatch = phase1Response.match(new RegExp(`${i + 1}[^0-9]*([0-9]{1,3})/100|Score[^0-9]*([0-9]{1,3})`, 'i'));
    currentScores[question] = scoreMatch ? parseInt(scoreMatch[1] || scoreMatch[2]) : 75;
  });
  
  // Phase 2 (Pushback)
  const phase2Prompt = buildPhase2Prompt(text, questions, currentScores, protocolType);
  const phase2Response = await callAIModel(model, phase2Prompt);
  responses.push(phase2Response);
  
  // Update scores after pushback
  questions.forEach((question, i) => {
    const scoreMatch = phase2Response.match(new RegExp(`${i + 1}[^0-9]*([0-9]{1,3})/100|Score[^0-9]*([0-9]{1,3})`, 'i'));
    if (scoreMatch) {
      currentScores[question] = parseInt(scoreMatch[1] || scoreMatch[2]);
    }
  });
  
  // Phase 3 (Walmart Metric)
  const phase3Prompt = buildPhase3Prompt(currentScores);
  const phase3Response = await callAIModel(model, phase3Prompt);
  responses.push(phase3Response);
  
  // Phase 4 (Final Validation)
  const phase4Prompt = buildPhase4Prompt(protocolType);
  const phase4Response = await callAIModel(model, phase4Prompt);
  responses.push(phase4Response);
  
  // Parse final scores
  questions.forEach((question, i) => {
    const scoreMatch = phase4Response.match(new RegExp(`${i + 1}[^0-9]*([0-9]{1,3})/100|Score[^0-9]*([0-9]{1,3})`, 'i'));
    if (scoreMatch) {
      currentScores[question] = parseInt(scoreMatch[1] || scoreMatch[2]);
    }
  });
  
  const finalScore = Math.round(Object.values(currentScores).reduce((a, b) => a + b, 0) / Object.values(currentScores).length);
  
  return {
    protocol: `comprehensive-${protocolType}`,
    mode: 'comprehensive',
    phase: 4,
    questions,
    responses,
    scores: currentScores,
    finalScore,
    summary: phase4Response.substring(0, 500) + '...',
    category: extractCategory(phase4Response)
  };
}

function extractCategory(response: string): string {
  // Look for category mentions in the response
  const categories = [
    'narcissistic', 'depressive', 'obsessional', 'resilient', 'fragmented',
    'neurotic', 'borderline', 'psychotic', 'genius', 'brilliant', 'mediocre'
  ];
  
  for (const category of categories) {
    if (response.toLowerCase().includes(category)) {
      return category;
    }
  }
  
  return 'unspecified';
}

// Main entry point for protocol analysis
export async function executeProtocolAnalysis(
  text: string,
  protocols: ProtocolType[],
  model: string = 'deepseek'
): Promise<ComprehensiveProtocolResult> {
  const startTime = Date.now();
  const results: ProtocolResult[] = [];
  
  for (const protocol of protocols) {
    try {
      if (protocol.startsWith('comprehensive-')) {
        const baseProtocol = protocol.replace('comprehensive-', '') as 'cognitive' | 'psychological' | 'psychopathological';
        const result = await executeComprehensiveProtocol(text, baseProtocol, model);
        results.push(result);
      } else {
        const result = await executeNormalProtocol(text, protocol as 'cognitive' | 'psychological' | 'psychopathological', model);
        results.push(result);
      }
    } catch (error) {
      console.error(`Failed to execute protocol ${protocol}:`, error);
      // Continue with other protocols even if one fails
    }
  }
  
  const totalTime = Date.now() - startTime;
  
  const overallSummary = generateOverallSummary(results);
  
  return {
    results,
    overallSummary,
    totalTime
  };
}

function generateOverallSummary(results: ProtocolResult[]): string {
  const avgScores = results.reduce((acc, result) => acc + result.finalScore, 0) / results.length;
  const categories = results.map(r => r.category).filter(Boolean);
  
  return `
## Overall Assessment

**Average Score**: ${Math.round(avgScores)}/100

**Identified Categories**: ${Array.from(new Set(categories)).join(', ')}

**Protocols Analyzed**: ${results.map(r => r.protocol).join(', ')}

**Key Insights**: 
${results.map(result => `
- **${result.protocol.toUpperCase()}**: Score ${result.finalScore}/100, Category: ${result.category}
`).join('')}

This comprehensive analysis provides multi-dimensional psychological profiling across cognitive, psychological, and psychopathological domains.
`;
}