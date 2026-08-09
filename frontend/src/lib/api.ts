export interface DecisionRates {
  observed: number;
  expected: number;
  shrunk: number;
}

export interface CoachProfile {
  coach: string;
  n_decisions: number;
  shrinkage_weight: number;
  last_season: number;
  punt: DecisionRates;
  field_goal: DecisionRates;
  go_for_it: DecisionRates;
}

export interface DecisionProbabilities {
  punt: number;
  field_goal: number;
  go_for_it: number;
}

export interface PredictResponse {
  predicted: DecisionProbabilities;
  coach_career_average: DecisionProbabilities;
  league_baseline: DecisionProbabilities;
  conversion_probability: number;
}

export interface SituationInput {
  coach: string;
  ydstogo: number;
  yardline_100: number;
  score_differential: number;
  game_seconds_remaining: number;
  qtr: number;
  posteam_timeouts_remaining: number;
  defteam_timeouts_remaining: number;
  is_home: boolean;
}

const API_URL = import.meta.env.PUBLIC_API_URL ?? "http://localhost:8000";

export async function getCoaches(): Promise<CoachProfile[]> {
  const response = await fetch(`${API_URL}/coaches`);
  if (!response.ok) throw new Error(`Failed to load coaches: ${response.status}`);
  return response.json();
}

export async function getCoach(name: string): Promise<CoachProfile> {
  const response = await fetch(`${API_URL}/coaches/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`Failed to load coach: ${response.status}`);
  return response.json();
}

export interface HealthInfo {
  name: string;
  latest_season: number;
  n_coaches: number;
}

export async function getHealth(): Promise<HealthInfo> {
  const response = await fetch(`${API_URL}/`);
  if (!response.ok) throw new Error(`Failed to load health info: ${response.status}`);
  return response.json();
}

export async function predict(situation: SituationInput): Promise<PredictResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(situation),
  });
  if (!response.ok) throw new Error(`Prediction failed: ${response.status}`);
  return response.json();
}

export interface SituationalSplit {
  distance_bucket: string;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface CoachBucketEntry {
  coach: string;
  distance_bucket: string;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface LeagueTrendPoint {
  season: number;
  n_decisions: number;
  go_for_it_rate: number;
}

export interface ConversionByDistance {
  distance_bucket: string;
  conversion_probability: number;
}

export interface FindingsData {
  situational_splits: SituationalSplit[];
  coach_bucket_leaderboard: CoachBucketEntry[];
  league_trend: LeagueTrendPoint[];
  conversion_by_distance: ConversionByDistance[];
}

export async function getFindings(): Promise<FindingsData> {
  const response = await fetch(`${API_URL}/findings`);
  if (!response.ok) throw new Error(`Failed to load findings: ${response.status}`);
  return response.json();
}
