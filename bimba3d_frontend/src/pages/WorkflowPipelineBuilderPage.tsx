import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";
import ConfirmModal from "../components/ConfirmModal";
import PipelineEditWarning from "../components/pipelineBuilder/PipelineEditWarning";
import PipelineBuilderHeader from "../components/pipelineBuilder/PipelineBuilderHeader";
import PipelineStepIndicator from "../components/pipelineBuilder/PipelineStepIndicator";

const API_BASE = "http://localhost:8005";
const DEFAULT_PIPELINE_MAX_STEPS = 7000;
const DEFAULT_GAUSSIAN_HARD_CAP = 6000000;
const LOG_SPACE_BOUND_PRESETS = {
 uniform: {
 geometry: [0.5, 2.0],
 appearance: [0.5, 2.0],
 densification: [0.7, 1.42],
 },
 mild: {
 geometry: [0.8, 1.25],
 appearance: [0.8, 1.25],
 densification: [0.85, 1.18],
 },
 gaussian: {
 geometry: [0.95, 1.05],
 appearance: [0.95, 1.05],
 densification: [0.95, 1.05],
 },
} as const;

interface DatasetInfo {
 name: string;
 path: string;
 image_count: number;
 size_mb: number;
 has_images: boolean;
 selected?: boolean;
 colmap_source_project_id?: string;
}

interface ExistingProject {
 id: string;
 name: string;
 has_colmap: boolean;
 dataset_path?: string;
 pipeline_name?: string;
}

interface PhaseConfig {
 phase_number: number;
 name: string;
 exploration_runs_per_project: number;
 strategy_override?: string;
 preset_override?: string;
 update_model: boolean;
 context_jitter: boolean;
 // run_jitter_only is derived automatically server-side: always true when context_jitter=true
 context_jitter_mode?: string;
 shuffle_order: boolean;
 session_execution_mode: string;
}


export default function WorkflowPipelineBuilderPage() {
 const navigate = useNavigate();
 const [searchParams] = useSearchParams();
 const editPipelineId = searchParams.get("edit");
 const returnTo = searchParams.get("returnTo");
 const requestedPipelineType = searchParams.get("type");
 const builderBackTarget =
 returnTo ||
 (editPipelineId
 ? `/workflow/pipelines/${editPipelineId}`
 : requestedPipelineType === "test"
 ? "/testing-pipeline"
 : requestedPipelineType === "offline_data"
 ? "/offline-data-preparation"
 : "/workflow");
 const lockedPipelineType: "offline_data" | "test" | null =
 requestedPipelineType === "test" ? "test" : requestedPipelineType === "offline_data" ? "offline_data" : null;
 const [isEditMode, setIsEditMode] = useState(false);
 const [loadingPipeline, setLoadingPipeline] = useState(false);
 const [loadedPipelineStatus, setLoadedPipelineStatus] = useState<string | null>(null);

 // Step 1: Dataset Selection
 const [baseDirectory, setBaseDirectory] = useState("E:\\Thesis\\Final Data\\Train_Test_Data\\Training Data");
 const [pipelineDirectory, setPipelineDirectory] = useState("E:\\Thesis\\PipelineProjects");
 const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
 const [scanning, setScanning] = useState(false);
 const [existingProjects, setExistingProjects] = useState<ExistingProject[]>([]);

 // Pipeline Type
 const [pipelineType, setPipelineType] = useState<"offline_data" | "test">(lockedPipelineType || "offline_data");
 const [sourceModelIds, setSourceModelIds] = useState<string[]>([]);
 const [availableModels, setAvailableModels] = useState<any[]>([]);

 // Step 2: Shared Configuration
 const [aiInputMode] = useState("exif_compact_featurewise");
 const aiSelectorStrategy = "featurewise_ridge_regression";
 const [maxSteps, setMaxSteps] = useState(DEFAULT_PIPELINE_MAX_STEPS);
 const [evalInterval, setEvalInterval] = useState(1000);
 const [logInterval, setLogInterval] = useState(100);
 const [densifyUntil, setDensifyUntil] = useState(4000);
 const [gaussianHardCap, setGaussianHardCap] = useState(DEFAULT_GAUSSIAN_HARD_CAP);
 const [imagesMaxSize, setImagesMaxSize] = useState(1600);
 const [geometryLogMin, setGeometryLogMin] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.geometry[0]);
 const [geometryLogMax, setGeometryLogMax] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.geometry[1]);
 const [appearanceLogMin, setAppearanceLogMin] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.appearance[0]);
 const [appearanceLogMax, setAppearanceLogMax] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.appearance[1]);
 const [densificationLogMin, setDensificationLogMin] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.densification[0]);
 const [densificationLogMax, setDensificationLogMax] = useState<number>(LOG_SPACE_BOUND_PRESETS.uniform.densification[1]);

 // Storage Management
 const [saveEvalImages, setSaveEvalImages] = useState(true);
 const [replaceEvalImages, setReplaceEvalImages] = useState(true); // Default: save storage
 const [saveCheckpoints, setSaveCheckpoints] = useState(true);
 const [replaceCheckpoints, setReplaceCheckpoints] = useState(true); // Default: save storage
 const [saveFinalSplat, setSaveFinalSplat] = useState(true); // Always recommended

 // Step 3: Training Schedule
 const [phases, setPhases] = useState<PhaseConfig[]>([
 {
 phase_number: 1,
 name: "Baseline Collection",
 exploration_runs_per_project: 1,
 preset_override: "balanced",
 update_model: false,
 context_jitter: false,
 context_jitter_mode: "uniform",
 shuffle_order: false,
 session_execution_mode: "test",
 },
 {
 phase_number: 2,
 name: "Exploration",
 exploration_runs_per_project: 5,
 update_model: false,
 context_jitter: true,
 context_jitter_mode: "uniform",
 shuffle_order: true,
 session_execution_mode: "train",
 },
 ]);

 // Step 4: Thermal Management
 const [thermalEnabled, setThermalEnabled] = useState(true);
 const [thermalStrategy, setThermalStrategy] = useState("fixed_interval");
 const [cooldownMinutes, setCooldownMinutes] = useState(10);

 // Step 5: Review
 const [pipelineName, setPipelineName] = useState(`Training_${new Date().toISOString().split("T")[0]}`);
 const [creating, setCreating] = useState(false);

 // UI State
 const [currentStep, setCurrentStep] = useState(1);
 const [showCreateConfirm, setShowCreateConfirm] = useState(false);
 const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

 const showToast = (message: string, type: "success" | "error" = "success") => {
 setToast({ message, type });
 window.setTimeout(() => setToast(null), 5000);
 };

 const errorMessage = (error: any, fallback: string) => {
 const detail = error?.response?.data?.detail;
 if (typeof detail === "string") return detail;
 if (detail && typeof detail.message === "string") return detail.message;
 if (error?.message) return error.message;
 return fallback;
 };

 const applyLogSpacePreset = (mode: keyof typeof LOG_SPACE_BOUND_PRESETS) => {
 const preset = LOG_SPACE_BOUND_PRESETS[mode];
 setGeometryLogMin(preset.geometry[0]);
 setGeometryLogMax(preset.geometry[1]);
 setAppearanceLogMin(preset.appearance[0]);
 setAppearanceLogMax(preset.appearance[1]);
 setDensificationLogMin(preset.densification[0]);
 setDensificationLogMax(preset.densification[1]);
 };

 const filterToAvailableModelIds = (selectedIds: string[], models: any[]) => {
 const visibleIds = new Set((models || []).map((m: any) => String(m.model_id || "")).filter(Boolean));
 return selectedIds.filter((id) => visibleIds.has(String(id)));
 };

 const visibleSelectedModelIds = filterToAvailableModelIds(sourceModelIds, availableModels);
 const staleSelectedModelIds = sourceModelIds.filter((id) => !visibleSelectedModelIds.includes(id));

 useEffect(() => {
 if (editPipelineId || lockedPipelineType !== "test") return;
 setPipelineType("test");
 setBaseDirectory("E:\\Thesis\\Final Data\\Train_Test_Data\\Test Data");
 setPipelineDirectory("E:\\Thesis\\PipelineTests");
 setPipelineName(`Test_${new Date().toISOString().split("T")[0]}`);
 setPhases([
 { phase_number: 1, name: "Baseline", exploration_runs_per_project: 1, preset_override: "balanced", update_model: false, context_jitter: false, context_jitter_mode: "uniform", shuffle_order: false, session_execution_mode: "train" },
 { phase_number: 2, name: "Model Test", exploration_runs_per_project: 1, update_model: false, context_jitter: false, context_jitter_mode: "uniform", shuffle_order: false, session_execution_mode: "test" },
 ]);
 axios.get(`${API_BASE}/api/models`).then((res) => {
 const models = res.data?.items || [];
 setAvailableModels(models);
 }).catch(() => setAvailableModels([]));
 }, [editPipelineId, lockedPipelineType]);

 // Load pipeline for editing
 useEffect(() => {
 if (editPipelineId) {
 setIsEditMode(true);
 setLoadingPipeline(true);
 axios.get(`${API_BASE}/api/workflow/pipelines/${editPipelineId}`)
 .then((response) => {
 const pipeline = response.data;
 const config = pipeline.config;

 // Load configuration
 setPipelineName(pipeline.name);
 setLoadedPipelineStatus(pipeline.status || null);
 setBaseDirectory(config.base_directory || "");
 setPipelineDirectory(config.pipeline_directory || "");

 // Pipeline type and test model selection
 const loadedType = config.pipeline_type === "train" ? "offline_data" : (config.pipeline_type || "offline_data");
 setPipelineType(loadedType as "offline_data" | "test");
 if (loadedType === "test") {
 const ids = config.source_model_ids || (config.source_model_id ? [config.source_model_id] : []);
 setSourceModelIds(ids);
 // Load available models so the checkboxes can render
 axios.get(`${API_BASE}/api/models`).then((res) => {
 const models = res.data?.items || [];
 setAvailableModels(models);
 const prunedIds = filterToAvailableModelIds(ids, models);
 if (prunedIds.length !== ids.length) {
 setSourceModelIds(prunedIds);
 showToast("Some previously selected test models no longer exist in elevated models and were removed from the editor selection. Save config to apply.", "error");
 }
 }).catch(() => setAvailableModels([]));
 }

 // Shared config
 const shared = config.shared_config || {};
 // aiSelectorStrategy is fixed to the report Ridge selector.
 setMaxSteps(shared.max_steps || DEFAULT_PIPELINE_MAX_STEPS);
 setEvalInterval(shared.eval_interval || 1000);
 setLogInterval(shared.log_interval || 100);
 setDensifyUntil(shared.densify_until_iter || 4000);
 setGaussianHardCap(shared.gaussian_hard_cap || DEFAULT_GAUSSIAN_HARD_CAP);
 setImagesMaxSize(shared.images_max_size || 1600);
 setGeometryLogMin(Number(shared.geometry_log_multiplier_min ?? LOG_SPACE_BOUND_PRESETS.uniform.geometry[0]));
 setGeometryLogMax(Number(shared.geometry_log_multiplier_max ?? LOG_SPACE_BOUND_PRESETS.uniform.geometry[1]));
 setAppearanceLogMin(Number(shared.appearance_log_multiplier_min ?? LOG_SPACE_BOUND_PRESETS.uniform.appearance[0]));
 setAppearanceLogMax(Number(shared.appearance_log_multiplier_max ?? LOG_SPACE_BOUND_PRESETS.uniform.appearance[1]));
 setDensificationLogMin(Number(shared.densification_log_multiplier_min ?? LOG_SPACE_BOUND_PRESETS.uniform.densification[0]));
 setDensificationLogMax(Number(shared.densification_log_multiplier_max ?? LOG_SPACE_BOUND_PRESETS.uniform.densification[1]));
 setSaveEvalImages(shared.save_eval_images !== false);
 setReplaceEvalImages(shared.replace_eval_images !== false);
 setSaveCheckpoints(shared.save_checkpoints !== false);
 setReplaceCheckpoints(shared.replace_checkpoints !== false);
 setSaveFinalSplat(shared.save_final_splat !== false);

 // Phases
 if (config.phases && config.phases.length > 0) {
 setPhases(config.phases.map((phase: any) => ({
 ...phase,
 exploration_runs_per_project: Number(phase.exploration_runs_per_project ?? 1),
 })));
 }

 // Thermal
 const thermal = config.thermal_management || {};
 setThermalEnabled(thermal.enabled !== false);
 setThermalStrategy(thermal.strategy || "fixed_interval");
 setCooldownMinutes(thermal.cooldown_minutes || 10);

 // Projects (datasets)
 if (config.projects && config.projects.length > 0) {
 const loadedDatasets = config.projects.map((p: any) => ({
 name: p.name,
 path: p.dataset_path,
 image_count: p.image_count || 0,
 size_mb: 0,
 has_images: true,
 selected: true,
 colmap_source_project_id: p.colmap_source_project_id,
 }));
 setDatasets(loadedDatasets);
 }

 showToast("Pipeline loaded for editing. Changes will require a restart to take effect.", "success");
 setCurrentStep(5); // Go to review step
 })
 .catch((error) => {
 console.error("Failed to load pipeline:", error);
 showToast(errorMessage(error, "Failed to load pipeline"), "error");
 navigate(builderBackTarget);
 })
 .finally(() => {
 setLoadingPipeline(false);
 });
 }
 }, [editPipelineId]);

 // Load existing projects with COLMAP
 const loadExistingProjects = async () => {
 try {
 // TEMP LEGACY: include old pipeline projects only so COLMAP sparse outputs can be copied once.
 const response = await axios.get(`${API_BASE}/projects`, {
 params: { include_legacy_colmap_sources: true },
 });
 const projects: ExistingProject[] = (response.data || []).map((p: any) => ({
 id: p.project_id,
 name: p.name || p.project_id,
 has_colmap: p.has_colmap || false,
 dataset_path: p.dataset_path,
 pipeline_name: p.pipeline_name,
 }));
 // Only show projects that have COLMAP outputs
 setExistingProjects(projects.filter(p => p.has_colmap));
 } catch (error: any) {
 console.error("Failed to load projects:", error);
 setExistingProjects([]);
 }
 };

 // Scan directory for datasets
 const handleScanDirectory = async () => {
 if (!baseDirectory.trim()) {
 alert("Please enter a directory path");
 return;
 }

 setScanning(true);
 try {
 const response = await axios.post(`${API_BASE}/api/workflow/pipelines/scan-directory`, {
 base_directory: baseDirectory,
 });

 const scannedDatasets = response.data.datasets.map((d: DatasetInfo) => ({
 ...d,
 selected: true, // Auto-select all by default
 colmap_source_project_id: undefined,
 }));

 setDatasets(scannedDatasets);

 // Load existing projects for COLMAP source selection
 await loadExistingProjects();
 } catch (error: any) {
 console.error("Failed to scan directory:", error);
 alert(`Failed to scan directory: ${errorMessage(error, "Failed to scan directory")}`);
 } finally {
 setScanning(false);
 }
 };

 // Calculate total runs
 const calculateTotalRuns = () => {
 const selectedCount = datasets.filter((d) => d.selected).length;
 const modelCount = pipelineType === "test" && sourceModelIds.length > 0 ? sourceModelIds.length : 1;
 let total = 0;
 for (const phase of phases) {
 if (pipelineType === "test" && phase.phase_number > 1) {
 total += phase.exploration_runs_per_project * selectedCount * modelCount;
 } else {
 total += phase.exploration_runs_per_project * selectedCount;
 }
 }
 return total;
 };

 // Calculate estimated time
 const calculateEstimatedTime = () => {
 const totalRuns = calculateTotalRuns();
 const trainingMinutes = totalRuns * 8; // Assume 8 minutes per run
 const cooldownTime = thermalEnabled ? totalRuns * cooldownMinutes : 0;
 const totalMinutes = trainingMinutes + cooldownTime;
 const hours = Math.floor(totalMinutes / 60);
 const minutes = totalMinutes % 60;
 return { hours, minutes, totalMinutes };
 };

 const modelFamilyLabel = (model: any) => {
 const family = String(model.model_family || model.artifact_format || model.ai_profile?.ai_selector_strategy || model.ai_profile?.ai_input_mode || "").toLowerCase();
 if (family.includes("compact") && family.includes("mlp")) return "Compact Featurewise MLP";
 if (family.includes("compact") && family.includes("ridge")) return "Compact Featurewise Ridge";
 if (family.includes("mlp")) return "Featurewise MLP";
 if (family.includes("ridge")) return "Featurewise Ridge";
 return "Workflow model";
 };

 const renderTestModelSelector = (compact = false) => {
 if (pipelineType !== "test") return null;
 return (
 <div style={{ marginBottom: compact ? "15px" : "15px", padding: "12px", background: "#f0f4ff", borderRadius: "6px", border: "1px solid #c7d2fe" }}>
 <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
 Models to Test (select one or more):
 </label>
 <div style={{ maxHeight: compact ? "170px" : "200px", overflowY: "auto", border: "1px solid #e2e8f0", borderRadius: "4px", padding: "8px", background: "#fff" }}>
 {availableModels.length === 0 ? (
 <p style={{ fontSize: "12px", color: "#666" }}>No workflow models found. Train featurewise or compact models from Training Data first.</p>
 ) : (
 availableModels.map((m: any) => (
 <label key={m.model_id} style={{ display: "flex", alignItems: "center", padding: "5px 0", fontSize: "13px", cursor: "pointer", gap: "8px" }}>
 <input
 type="checkbox"
 checked={sourceModelIds.includes(m.model_id)}
 onChange={(e) => {
 if (e.target.checked) {
 setSourceModelIds((prev) => Array.from(new Set([...prev, m.model_id])));
 } else {
 setSourceModelIds((prev) => prev.filter((id) => id !== m.model_id));
 }
 }}
 />
 <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={m.model_name || m.model_id}>
 {m.model_name || m.model_id}
 </span>
 <span style={{ flexShrink: 0, fontSize: "11px", color: "#4338ca", background: "#eef2ff", border: "1px solid #c7d2fe", borderRadius: "999px", padding: "1px 7px" }}>
 {modelFamilyLabel(m)}
 </span>
 </label>
 ))
 )}
 </div>
 {visibleSelectedModelIds.length > 0 && (
 <p style={{ fontSize: "12px", color: "#4338ca", marginTop: "5px" }}>
 {visibleSelectedModelIds.length} model(s) selected; each test project will run one baseline and one AI-guided run per selected model.
 </p>
 )}
 {staleSelectedModelIds.length > 0 && (
 <p style={{ fontSize: "12px", color: "#b45309", marginTop: "5px" }}>
 {staleSelectedModelIds.length} stale saved model selection(s) are still present in backend config but not matched to the visible list. Saving this config will remove them.
 </p>
 )}
 </div>
 );
 };

 // Show create confirmation
 const handleShowCreateConfirm = () => {
 const selectedDatasets = datasets.filter((d) => d.selected);

 if (isEditMode && loadedPipelineStatus?.toLowerCase() === "running") {
 showToast("Stop or pause this pipeline before saving configuration changes.", "error");
 return;
 }

 if (selectedDatasets.length === 0) {
 alert("Please select at least one dataset");
 return;
 }

 if (pipelineType === "test" && visibleSelectedModelIds.length === 0) {
 alert("Please select at least one trained workflow model to test.");
 return;
 }

 setShowCreateConfirm(true);
 };

 // Create pipeline (after confirmation)
 const handleCreatePipeline = async () => {
 setShowCreateConfirm(false);
 setCreating(true);
 try {
 const selectedDatasets = datasets.filter((d) => d.selected);
 const validSelectedModelIds = filterToAvailableModelIds(sourceModelIds, availableModels);
 setSourceModelIds(validSelectedModelIds);

 // Build configuration
 const config = {
 name: pipelineName,
 base_directory: baseDirectory,
 pipeline_directory: pipelineDirectory || null, // null = use default
 pipeline_type: pipelineType,
 source_model_id:
 pipelineType === "test"
 ? (validSelectedModelIds[0] || null)
 : null,
 source_model_ids:
 pipelineType === "test"
 ? (validSelectedModelIds.length > 0 ? validSelectedModelIds : null)
 : null,
 contribute_to_training: false, // Test pipelines never update model; use project-level test for that
 projects: selectedDatasets.map((d) => ({
 name: d.name,
 dataset_path: d.path,
 image_count: d.image_count,
 created: false,
 colmap_source_project_id: d.colmap_source_project_id || null,
 })),
 shared_config: {
 ai_input_mode: aiInputMode,
 ai_selector_strategy: aiSelectorStrategy,
 max_steps: maxSteps,
 eval_interval: evalInterval,
 log_interval: logInterval,
 densify_until_iter: densifyUntil,
 gaussian_hard_cap: gaussianHardCap,
 images_max_size: imagesMaxSize,
 geometry_log_multiplier_min: geometryLogMin,
 geometry_log_multiplier_max: geometryLogMax,
 appearance_log_multiplier_min: appearanceLogMin,
 appearance_log_multiplier_max: appearanceLogMax,
 densification_log_multiplier_min: densificationLogMin,
 densification_log_multiplier_max: densificationLogMax,
 // Storage management options
 save_eval_images: saveEvalImages,
 replace_eval_images: replaceEvalImages,
 save_checkpoints: saveCheckpoints,
 replace_checkpoints: replaceCheckpoints,
 save_final_splat: saveFinalSplat,
 },
 phases: phases,
 thermal_management: {
 enabled: thermalEnabled,
 strategy: thermalStrategy,
 cooldown_minutes: cooldownMinutes,
 gpu_temp_threshold: 70,
 check_interval_seconds: 30,
 max_wait_minutes: 30,
 },
 failure_handling: {
 continue_on_failure: true,
 max_retries_per_run: 1,
 skip_project_after_failures: 3,
 },
 };

 // Create or update pipeline
 if (isEditMode && editPipelineId) {
 const res = await axios.put(`${API_BASE}/api/workflow/pipelines/${editPipelineId}/config`, config);
 const responseMessage = typeof res.data?.message === "string" ? res.data.message : "Configuration updated.";
 showToast(`Pipeline "${pipelineName}" updated successfully! ${responseMessage}`, "success");
 setTimeout(() => {
 navigate(builderBackTarget || `/workflow/pipelines/${editPipelineId}`);
 }, 1500);
 } else {
 await axios.post(`${API_BASE}/api/workflow/pipelines`, config);
 showToast(`Pipeline "${pipelineName}" created successfully!`, "success");
 setTimeout(() => {
 navigate(builderBackTarget);
 }, 1500);
 }

 } catch (error: any) {
 console.error(isEditMode ? "Failed to update pipeline:" : "Failed to create pipeline:", error);
 showToast(
 `${isEditMode ? "Failed to update pipeline" : "Failed to create pipeline"}: ${errorMessage(error, "Request failed")}`,
 "error"
 );
 } finally {
 setCreating(false);
 }
 };

 if (loadingPipeline) {
 return (
 <div className="flex items-center justify-center h-screen">
 <div className="text-center">
 <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
 <p className="text-gray-600">Loading pipeline configuration...</p>
 </div>
 </div>
 );
 }

 return (
 <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
 <PipelineBuilderHeader
 isEditMode={isEditMode}
 pipelineType={pipelineType}
 onBack={() => navigate(builderBackTarget)}
 />

 {isEditMode && <PipelineEditWarning />}

 <PipelineStepIndicator currentStep={currentStep} pipelineType={pipelineType} />

 {/* Main Content */}
 <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">

 {/* Step 1: Pipeline Setup */}
 {currentStep === 1 && (
 <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
 <h2>Step 1: Pipeline Setup</h2>

 {/* Pipeline Type Toggle */}
 <div style={{ marginBottom: "20px", display: "flex", gap: "10px", alignItems: "center" }}>
 <label style={{ fontWeight: "bold" }}>Workflow Stage:</label>
 {(!lockedPipelineType || lockedPipelineType === "offline_data") && (
 <button
 onClick={() => {
 setPipelineType("offline_data");
 setBaseDirectory("E:\\Thesis\\Final Data\\Train_Test_Data\\Training Data");
 setPipelineDirectory("E:\\Thesis\\PipelineProjects");
 setPipelineName(`Preparation_${new Date().toISOString().split("T")[0]}`);
 setPhases([
 { phase_number: 1, name: "Baseline Collection", exploration_runs_per_project: 1, preset_override: "balanced", update_model: false, context_jitter: false, context_jitter_mode: "uniform", shuffle_order: false, session_execution_mode: "test" },
 { phase_number: 2, name: "Exploration", exploration_runs_per_project: 5, update_model: false, context_jitter: true, context_jitter_mode: "uniform", shuffle_order: true, session_execution_mode: "train" },
 ]);
 }}
 style={{
 padding: "6px 16px",
 borderRadius: "4px",
 border: "1px solid #4f46e5",
 background: pipelineType === "offline_data" ? "#4f46e5" : "white",
 color: pipelineType === "offline_data" ? "white" : "#4f46e5",
 fontWeight: "bold",
 cursor: "pointer",
 }}
 >
 Offline Preparation
 </button>
 )}
 {(!lockedPipelineType || lockedPipelineType === "test") && (
 <button
 onClick={() => {
 setPipelineType("test");
 setBaseDirectory("E:\\Thesis\\Final Data\\Train_Test_Data\\Test Data");
 setPipelineDirectory("E:\\Thesis\\PipelineTests");
 setPipelineName(`Test_${new Date().toISOString().split("T")[0]}`);
 // Test pipeline: only 2 phases (baseline + test)
 setPhases([
 { phase_number: 1, name: "Baseline", exploration_runs_per_project: 1, preset_override: "balanced", update_model: false, context_jitter: false, context_jitter_mode: "uniform", shuffle_order: false, session_execution_mode: "train" },
 { phase_number: 2, name: "Model Test", exploration_runs_per_project: 1, update_model: false, context_jitter: false, context_jitter_mode: "uniform", shuffle_order: false, session_execution_mode: "test" },
 ]);
 // Load available models
 axios.get(`${API_BASE}/api/models`).then((res) => {
 const models = res.data?.items || [];
 setAvailableModels(models);
 }).catch(() => setAvailableModels([]));
 }}
 style={{
 padding: "6px 16px",
 borderRadius: "4px",
 border: "1px solid #4f46e5",
 background: pipelineType === "test" ? "#4f46e5" : "white",
 color: pipelineType === "test" ? "white" : "#4f46e5",
 fontWeight: "bold",
 cursor: "pointer",
 }}
 >
 Testing Pipeline
 </button>
 )}
 </div>

 {renderTestModelSelector()}

 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
 Pipeline Name:
 </label>
 <input
 type="text"
 value={pipelineName}
 onChange={(e) => setPipelineName(e.target.value)}
 placeholder={pipelineType === "test" ? "my_testing_pipeline" : "my_preparation_pipeline"}
 style={{ width: "100%", padding: "8px" }}
 />
 <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
 Name for this {pipelineType === "test" ? "testing" : "offline preparation"} pipeline (will be used as the folder name)
 </p>
 </div>

 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
 Source Data Directory (Read-Only):
 </label>
 <div style={{ display: "flex", gap: "10px" }}>
 <input
 type="text"
 value={baseDirectory}
 onChange={(e) => setBaseDirectory(e.target.value)}
 placeholder="E:/Thesis/exp_new_method"
 style={{ flex: 1, padding: "8px" }}
 />
 <button onClick={handleScanDirectory} disabled={scanning} style={{ padding: "8px 16px" }}>
 {scanning ? "Scanning..." : "Scan Directory"}
 </button>
 </div>
 <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
 Directory containing dataset folders with images (will NOT be modified)
 </p>
 </div>

 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px", fontWeight: "bold" }}>
 Pipeline Output Directory:
 </label>
 <input
 type="text"
 value={pipelineDirectory}
 onChange={(e) => setPipelineDirectory(e.target.value)}
 placeholder="Leave empty to use default projects directory"
 style={{ width: "100%", padding: "8px" }}
 />
 <p style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}>
 Where to create the pipeline folder. Leave empty to use default location (same as manual projects).
 Pipeline will create: {pipelineDirectory || "[default]"}/{pipelineName}/
 </p>
 </div>

 {datasets.length > 0 && (
 <div>
 <h3>Discovered Datasets ({datasets.filter((d) => d.selected).length}/{datasets.length} selected):</h3>

 <div style={{ marginBottom: "10px" }}>
 <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: true })))} style={{ marginRight: "10px" }}>
 Select All
 </button>
 <button onClick={() => setDatasets(datasets.map((d) => ({ ...d, selected: false })))}>
 Deselect All
 </button>
 </div>

 <div style={{ maxHeight: "400px", overflowY: "auto", border: "1px solid #ddd", padding: "10px" }}>
 {datasets.map((dataset, idx) => (
 <div key={idx} style={{ padding: "8px", borderBottom: "1px solid #eee" }}>
 <div style={{ display: "flex", alignItems: "center", marginBottom: "8px" }}>
 <input
 type="checkbox"
 checked={dataset.selected}
 onChange={(e) => {
 const updated = [...datasets];
 updated[idx].selected = e.target.checked;
 setDatasets(updated);
 }}
 style={{ marginRight: "10px" }}
 />
 <div style={{ flex: 1 }}>
 <strong>{dataset.name}</strong>
 <div style={{ fontSize: "12px", color: "#666" }}>
 Images: {dataset.image_count} | Size: {dataset.size_mb.toFixed(1)} MB
 </div>
 </div>
 </div>
 {dataset.selected && (
 <div style={{ marginLeft: "30px", marginTop: "4px" }}>
 <label style={{ fontSize: "12px", color: "#555", display: "block", marginBottom: "4px" }}>
 Copy COLMAP from existing project (optional):
 </label>
 <select
 value={dataset.colmap_source_project_id || ""}
 onChange={(e) => {
 const updated = [...datasets];
 updated[idx].colmap_source_project_id = e.target.value || undefined;
 setDatasets(updated);
 }}
 style={{ width: "100%", padding: "4px", fontSize: "12px" }}
 >
 <option value="">-- Run COLMAP for this dataset --</option>
 {existingProjects.map((proj) => (
 <option key={proj.id} value={proj.id}>
 {proj.name}{proj.pipeline_name ? `, ${proj.pipeline_name}` : ''}
 </option>
 ))}
 </select>
 {dataset.colmap_source_project_id && (() => {
 const selectedProject = existingProjects.find(p => p.id === dataset.colmap_source_project_id);
 const datasetName = dataset.name.toLowerCase();
 const projectName = selectedProject?.name?.toLowerCase() || '';
 // Check if names match (allowing for space/underscore differences)
 const datasetNameNormalized = datasetName.replace(/\s+/g, '_');
 const projectNameNormalized = projectName.replace(/\s+/g, '_');
 const namesMatch = projectNameNormalized === datasetNameNormalized;

 return (
 <>
 <div style={{ fontSize: "11px", color: "#0066cc", marginTop: "2px" }}>
 Will copy COLMAP from selected project (saves ~15-30 min)
 </div>
 {!namesMatch && selectedProject && (
 <div style={{ fontSize: "11px", color: "#ff6600", marginTop: "2px", fontWeight: "bold" }}>
 Warning: Project name "{selectedProject.name}" does not match dataset name "{dataset.name}". COLMAP data may be from different images.
 </div>
 )}
 </>
 );
 })()}
 </div>
 )}
 </div>
 ))}
 </div>
 </div>
 )}

 <div style={{ marginTop: "20px", textAlign: "right" }}>
 <button onClick={() => setCurrentStep(2)} disabled={datasets.filter((d) => d.selected).length === 0} style={{ padding: "8px 24px" }}>
 Next
 </button>
 </div>
 </div>
 )}

 {/* Step 2: Shared Run Configuration */}
 {currentStep === 2 && (
 <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
 <h2>Step 2: Shared {pipelineType === "test" ? "Testing" : "Preparation"} Run Configuration</h2>

 <div style={{ marginBottom: "15px", padding: "10px", background: "#f0f4ff", border: "1px solid #b3c2f0", borderRadius: "4px", fontSize: "13px" }}>
 <strong>Selector Strategy:</strong> {pipelineType === "test" ? "Model-specific AI profile" : "Featurewise Ridge Regression by default"}
 <div style={{ fontSize: "11px", color: "#555", marginTop: "4px" }}>
 {pipelineType === "test"
 ? "Each selected model applies its own saved AI profile and candidate multiplier grid during testing."
 : "Predicts one multiplier per parameter group (geometry, appearance, densification) using project scene descriptors. Model training happens after offline data preparation."}
 </div>
 </div>

 {renderTestModelSelector(true)}

 <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px" }}>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>Max Steps:</label>
 <input type="number" value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
 </div>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>
 Eval Interval:
 <span style={{ fontSize: "11px", color: "#666", fontWeight: "normal", marginLeft: "5px" }}>
 (Quality evaluation frequency)
 </span>
 </label>
 <input
 type="number"
 value={evalInterval}
 onChange={(e) => setEvalInterval(Number(e.target.value))}
 style={{ width: "100%", padding: "8px" }}
 min={100}
 max={50000}
 step={100}
 />
 <p style={{ fontSize: "11px", color: "#888", marginTop: "3px" }}>
 More frequent = better quality tracking. Default: 1000
 </p>
 </div>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>Log Interval:</label>
 <input type="number" value={logInterval} onChange={(e) => setLogInterval(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
 </div>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>Densify Until:</label>
 <input type="number" value={densifyUntil} onChange={(e) => setDensifyUntil(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
 </div>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>
 Gaussian Hard Cap:
 <span style={{ fontSize: "11px", color: "#666", fontWeight: "normal", marginLeft: "5px" }}>
 (Stops a run when gaussian count exceeds this)
 </span>
 </label>
 <input
 type="number"
 value={gaussianHardCap}
 onChange={(e) => setGaussianHardCap(Math.max(1, Number(e.target.value)))}
 min={1}
 step={100000}
 style={{ width: "100%", padding: "8px" }}
 />
 <p style={{ fontSize: "11px", color: "#888", marginTop: "3px" }}>
 Default: {DEFAULT_GAUSSIAN_HARD_CAP.toLocaleString()}. Increase before retrying hard-cap runs if needed.
 </p>
 </div>
 <div>
 <label style={{ display: "block", marginBottom: "5px" }}>Images Max Size:</label>
 <input type="number" value={imagesMaxSize} onChange={(e) => setImagesMaxSize(Number(e.target.value))} style={{ width: "100%", padding: "8px" }} />
 </div>
 </div>

 {/* Storage Management Options */}
 <div style={{ marginTop: "25px", padding: "15px", background: "#fff8dc", border: "1px solid #daa520", borderRadius: "4px" }}>
 <h3 style={{ margin: "0 0 15px 0", fontSize: "16px", color: "#b8860b" }}>
 Storage Management
 </h3>
 <p style={{ fontSize: "12px", color: "#666", marginBottom: "15px" }}>
 Configure what gets saved to manage storage. Eval images at 200-500 step intervals can create massive storage requirements.
 </p>

 <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "12px" }}>
 {/* Save Eval Images */}
 <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
 <input
 type="checkbox"
 checked={saveEvalImages}
 onChange={(e) => setSaveEvalImages(e.target.checked)}
 style={{ width: "18px", height: "18px" }}
 />
 <span>
 <strong>Save Evaluation Images</strong>
 <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
 (Renders at each eval_interval. WARNING: ~5-50MB per eval num_evals = GBs)
 </span>
 </span>
 </label>

 {/* Replace Eval Images */}
 {saveEvalImages && (
 <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", marginLeft: "30px" }}>
 <input
 type="checkbox"
 checked={replaceEvalImages}
 onChange={(e) => setReplaceEvalImages(e.target.checked)}
 style={{ width: "18px", height: "18px" }}
 />
 <span>
 <strong>Replace Eval Images</strong> (Keep only latest eval, delete previous)
 <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
 (Saves ~95% storage. Use for preparation pipelines, disable for final runs)
 </span>
 </span>
 </label>
 )}

 {/* Save Checkpoints */}
 <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
 <input
 type="checkbox"
 checked={saveCheckpoints}
 onChange={(e) => setSaveCheckpoints(e.target.checked)}
 style={{ width: "18px", height: "18px" }}
 />
 <span>
 <strong>Save Training Checkpoints</strong>
 <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
 (Model weights for resuming. ~100-500MB per checkpoint)
 </span>
 </span>
 </label>

 {/* Replace Checkpoints */}
 {saveCheckpoints && (
 <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px", marginLeft: "30px" }}>
 <input
 type="checkbox"
 checked={replaceCheckpoints}
 onChange={(e) => setReplaceCheckpoints(e.target.checked)}
 style={{ width: "18px", height: "18px" }}
 />
 <span>
 <strong>Replace Checkpoints</strong> (Keep only latest checkpoint)
 <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
 (Recommended for pipelines. Keep only final model)
 </span>
 </span>
 </label>
 )}

 {/* Save Final Splat */}
 <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "14px" }}>
 <input
 type="checkbox"
 checked={saveFinalSplat}
 onChange={(e) => setSaveFinalSplat(e.target.checked)}
 style={{ width: "18px", height: "18px" }}
 />
 <span>
 <strong>Save Final Splat Model</strong>
 <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>
 (Always recommended. ~50-200MB. Needed for viewing results)
 </span>
 </span>
 </label>
 </div>
 </div>

 <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
 <button onClick={() => setCurrentStep(1)} style={{ padding: "8px 24px" }}>
 Back
 </button>
 <button
 onClick={() => setCurrentStep(3)}
 disabled={pipelineType === "test" && visibleSelectedModelIds.length === 0}
 style={{
 padding: "8px 24px",
 opacity: pipelineType === "test" && visibleSelectedModelIds.length === 0 ? 0.55 : 1,
 cursor: pipelineType === "test" && visibleSelectedModelIds.length === 0 ? "not-allowed" : "pointer",
 }}
 >
 Next
 </button>
 </div>
 </div>
 )}

 {/* Step 3: Preparation or Testing Schedule */}
 {currentStep === 3 && (
 <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
 <h2>Step 3: {pipelineType === "test" ? "Testing" : "Offline Preparation"} Schedule</h2>

 {/* Schedule explanation */}
 <div style={{ marginBottom: "20px", padding: "14px", background: "#f0f4ff", border: "1px solid #b3c2f0", borderRadius: "6px", fontSize: "12px", lineHeight: "1.7" }}>
 <strong style={{ fontSize: "13px" }}>How the schedule works</strong>
 <div style={{ marginTop: "8px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
 <div style={{ background: "#fff", padding: "10px", borderRadius: "4px", border: "1px solid #dce3f8" }}>
 <div style={{ fontWeight: "bold", color: "#3b4fd8", marginBottom: "4px" }}> Phase</div>
 <div>A named stage of the pipeline (e.g. Baseline Collection, Exploration). Each phase has its own log-multiplier and strategy settings. Phases run in order; all projects complete a phase before the next phase begins.</div>
 </div>
 <div style={{ background: "#fff", padding: "10px", borderRadius: "4px", border: "1px solid #dce3f8" }}>
 <div style={{ fontWeight: "bold", color: "#3b4fd8", marginBottom: "4px" }}> Exploration Runs</div>
 <div>How many times each project runs within a phase. Each run independently samples a new multiplier in log space, producing one training example (features + multipliers + score). More runs = more diverse training data for the offline model.</div>
 </div>
 </div>
 <div style={{ marginTop: "10px", padding: "8px", background: "#fffbe6", border: "1px solid #f0d080", borderRadius: "4px" }}>
 <strong>Example:</strong> 10 projects x 5 exploration runs = <strong>50 training examples</strong> in that phase.
 Each run samples a multiplier in log space (document 4.X.2).
 </div>
 <div style={{ marginTop: "8px", padding: "8px", background: "#e8f8e8", border: "1px solid #a0d8a0", borderRadius: "4px" }}>
 <strong>For data collection (Stage 1):</strong> Phase 1 = 1 baseline run per project (log-multipliers off, default params). Phase 2 = N exploration runs with log-space multipliers directly on base parameters; no model prediction. Each run becomes one training example for offline model training.
 </div>
 </div>

 {phases.map((phase, idx) => (
 <div key={idx} style={{ marginBottom: "20px", padding: "15px", background: "#f9f9f9", borderRadius: "4px" }}>
 <h3>Phase {phase.phase_number}: {phase.name}</h3>
 <p style={{ marginTop: "4px", marginBottom: "10px", fontSize: "11px", color: "#666" }}>
 {phase.phase_number === 2
 ? "Exploration phase: defaults to log multipliers on and log-multiplier-only, so the model samples parameter space directly without multiplying selected values."
 : phase.phase_number === 3
 ? "Learning phase: uses model-selected multipliers first, with optional small Gaussian perturbation if you want a light check against the selected values."
 : "Baseline phase: collect the reference run used for baseline-relative scoring."}
 </p>

 <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "10px" }}>
 <div>
 <label style={{ display: "block", fontSize: "12px", marginBottom: "3px" }}>
 {phase.phase_number === 1 ? "Runs (baseline always 1):" : "Exploration runs per project:"}
 </label>
 <input
 type="number"
 value={phase.exploration_runs_per_project}
 min={1}
 disabled={phase.phase_number === 1}
 onChange={(e) => {
 const updated = [...phases];
 updated[idx].exploration_runs_per_project = Math.max(1, Number(e.target.value));
 setPhases(updated);
 }}
 style={{ width: "100%", padding: "6px", background: phase.phase_number === 1 ? "#f5f5f5" : undefined }}
 />
 <div style={{ fontSize: "10px", color: "#888", marginTop: "3px" }}>
 {phase.phase_number === 1
 ? "One baseline run per project. Each run produces the reference metrics for score calculation."
 : `Each project trains ${phase.exploration_runs_per_project} time${phase.exploration_runs_per_project !== 1 ? "s" : ""} in this phase. Each run samples a new multiplier in log space one training example.`}
 </div>
 </div>
 </div>

 {/* Exploration Log Multipliers and Shuffle Settings */}
 <div style={{ marginTop: "10px", padding: "10px", background: "#fff", border: "1px solid #ddd", borderRadius: "4px" }}>
 {pipelineType === "offline_data" && (
 <>
 <div style={{ marginBottom: "8px" }}>
 <label style={{ display: "flex", alignItems: "center", fontSize: "12px" }}>
 <input
 type="checkbox"
 checked={phase.context_jitter}
 onChange={(e) => {
 const updated = [...phases];
 updated[idx].context_jitter = e.target.checked;
 setPhases(updated);
 }}
 style={{ marginRight: "8px" }}
 />
 <strong>Enable Exploration Log Multipliers</strong>
 <span style={{ marginLeft: "8px", color: "#666", fontWeight: "normal" }}>
 (Toggle exploration log multipliers on/off for this phase)
 </span>
 </label>
 </div>

 {phase.context_jitter && (
 <div style={{ marginTop: "8px" }}>
 <label style={{ display: "block", fontSize: "12px", marginBottom: "4px" }}>
 <strong>Log-Multiplier Mode:</strong>
 </label>
 <select
 value={phase.context_jitter_mode || "uniform"}
 onChange={(e) => {
 const updated = [...phases];
 updated[idx].context_jitter_mode = e.target.value;
 setPhases(updated);
 applyLogSpacePreset(e.target.value as keyof typeof LOG_SPACE_BOUND_PRESETS);
 }}
 style={{ width: "100%", padding: "6px", fontSize: "12px", borderRadius: "4px", border: "1px solid #ccc" }}
 >
 <option value="uniform">Uniform: geometry/appearance [0.5, 2.0], densification [0.7, 1.42]</option>
 <option value="mild">Mild: geometry/appearance [0.8, 1.25], densification [0.85, 1.18]</option>
 <option value="gaussian">Narrow: all groups [0.95, 1.05]</option>
 </select>
 <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "8px", marginTop: "8px" }}>
 <div style={{ border: "1px solid #ddd", borderRadius: "4px", padding: "8px", background: "#fff" }}>
 <div style={{ fontSize: "11px", fontWeight: 700, marginBottom: "6px" }}>Geometry</div>
 <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
 <input type="number" step="0.01" min="0.000001" value={geometryLogMin} onChange={(e) => setGeometryLogMin(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Geometry log-space minimum multiplier" />
 <input type="number" step="0.01" min="0.000001" value={geometryLogMax} onChange={(e) => setGeometryLogMax(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Geometry log-space maximum multiplier" />
 </div>
 </div>
 <div style={{ border: "1px solid #ddd", borderRadius: "4px", padding: "8px", background: "#fff" }}>
 <div style={{ fontSize: "11px", fontWeight: 700, marginBottom: "6px" }}>Appearance</div>
 <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
 <input type="number" step="0.01" min="0.000001" value={appearanceLogMin} onChange={(e) => setAppearanceLogMin(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Appearance log-space minimum multiplier" />
 <input type="number" step="0.01" min="0.000001" value={appearanceLogMax} onChange={(e) => setAppearanceLogMax(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Appearance log-space maximum multiplier" />
 </div>
 </div>
 <div style={{ border: "1px solid #ddd", borderRadius: "4px", padding: "8px", background: "#fff" }}>
 <div style={{ fontSize: "11px", fontWeight: 700, marginBottom: "6px" }}>Densification</div>
 <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
 <input type="number" step="0.01" min="0.000001" value={densificationLogMin} onChange={(e) => setDensificationLogMin(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Densification log-space minimum multiplier" />
 <input type="number" step="0.01" min="0.000001" value={densificationLogMax} onChange={(e) => setDensificationLogMax(Number(e.target.value))} style={{ width: "100%", padding: "5px", fontSize: "11px", border: "1px solid #ccc", borderRadius: "4px" }} title="Densification log-space maximum multiplier" />
 </div>
 </div>
 </div>
 </div>
 )}

 {phase.context_jitter && (
 <div style={{ marginTop: "10px", padding: "8px", background: "#f0f8ff", border: "1px solid #ccc", borderRadius: "4px", fontSize: "9px", color: "#333" }}>
 <strong>Exploration log multipliers (Stage 1 data collection):</strong> Each run samples one multiplier in log space and applies it directly to the default parameters. No model prediction is used predictions only happen after offline training (Stage 2).<br /><br />
 <strong>Log-multiplier ranges:</strong> Values are sampled log-uniformly within the configured group bounds, then saved in the pipeline config and reused after stop/resume.
 </div>
 )}

 <div style={{ marginTop: "8px" }}>
 <label style={{ display: "flex", alignItems: "center", fontSize: "12px" }}>
 <input
 type="checkbox"
 checked={phase.shuffle_order}
 onChange={(e) => {
 const updated = [...phases];
 updated[idx].shuffle_order = e.target.checked;
 setPhases(updated);
 }}
 style={{ marginRight: "8px" }}
 />
 <strong>Shuffle Project Order</strong>
 <span style={{ marginLeft: "8px", color: "#666", fontWeight: "normal" }}>
 (Randomize run sequence)
 </span>
 </label>
 </div>
 </>
 )}

 </div>
 </div>
 ))}

 <div style={{ padding: "15px", background: "#e3f2fd", borderRadius: "4px" }}>
 <strong>Grand Total: {calculateTotalRuns()} runs</strong>
 </div>

 <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
 <button onClick={() => setCurrentStep(2)} style={{ padding: "8px 24px" }}>
 Back
 </button>
 <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
 Next
 </button>
 </div>
 </div>
 )}

 {/* Step 4: Execution Management */}
 {currentStep === 4 && (
 <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
 <h2>Step 4: Execution Management</h2>

 <div style={{ marginBottom: "15px" }}>
 <label>
 <input type="checkbox" checked={thermalEnabled} onChange={(e) => setThermalEnabled(e.target.checked)} style={{ marginRight: "8px" }} />
 Enable cooldown periods between runs
 </label>
 </div>

 {thermalEnabled && (
 <div>
 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px" }}>Cooldown Strategy:</label>
 <select value={thermalStrategy} onChange={(e) => setThermalStrategy(e.target.value)} style={{ width: "100%", padding: "8px" }}>
 <option value="fixed_interval">Fixed Interval</option>
 <option value="temperature_based">Temperature-based (requires GPU monitoring)</option>
 <option value="time_scheduled">Time-of-day scheduling</option>
 </select>
 </div>

 {thermalStrategy === "fixed_interval" && (
 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px" }}>Wait time (minutes):</label>
 <input
 type="number"
 value={cooldownMinutes}
 onChange={(e) => setCooldownMinutes(Number(e.target.value))}
 style={{ width: "100%", padding: "8px" }}
 />
 </div>
 )}

 <div style={{ padding: "15px", background: "#fff3cd", borderRadius: "4px" }}>
 <h4>Estimated Total Time:</h4>
 <div>Training time: {calculateTotalRuns()} runs 8 min {Math.floor((calculateTotalRuns() * 8) / 60)} hours</div>
 <div>Cooldown time: {calculateTotalRuns()} {cooldownMinutes} min {Math.floor((calculateTotalRuns() * cooldownMinutes) / 60)} hours</div>
 <div style={{ fontWeight: "bold", marginTop: "10px" }}>
 Total: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m (~{(calculateEstimatedTime().totalMinutes / 1440).toFixed(1)} days)
 </div>
 </div>
 </div>
 )}

 <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
 <button onClick={() => setCurrentStep(3)} style={{ padding: "8px 24px" }}>
 Back
 </button>
 <button onClick={() => setCurrentStep(5)} style={{ padding: "8px 24px" }}>
 Next
 </button>
 </div>
 </div>
 )}

 {/* Step 5: Review & Launch */}
 {currentStep === 5 && (
 <div style={{ border: "1px solid #ddd", padding: "20px", borderRadius: "4px", marginBottom: "20px" }}>
 <h2>Step 5: Review & Launch</h2>

 <div style={{ padding: "15px", background: "#f5f5f5", borderRadius: "4px", marginBottom: "15px" }}>
 <h3>Pipeline Summary:</h3>
 <ul>
 <li>Projects: {datasets.filter((d) => d.selected).length}</li>
 <li>Total runs: {calculateTotalRuns()}</li>
 <li>Strategy: {aiSelectorStrategy}</li>
 <li>Estimated duration: ~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</li>
 </ul>
 </div>

 <div style={{ marginBottom: "15px" }}>
 <label style={{ display: "block", marginBottom: "5px" }}>Pipeline Name:</label>
 <input
 type="text"
 value={pipelineName}
 onChange={(e) => setPipelineName(e.target.value)}
 style={{ width: "100%", padding: "8px" }}
 />
 </div>

 <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between" }}>
 <button onClick={() => setCurrentStep(4)} style={{ padding: "8px 24px" }}>
 Back
 </button>
 <button
 onClick={handleShowCreateConfirm}
 disabled={creating}
 style={{ padding: "8px 24px", background: "#4CAF50", color: "white", fontWeight: "bold", border: "none", borderRadius: "4px" }}
 >
 {creating
 ? (isEditMode ? "Updating..." : "Creating...")
 : (isEditMode ? "Save Changes" : "Create Pipeline")}
 </button>
 </div>
 </div>
 )}

 <ConfirmModal
 open={showCreateConfirm}
 title={isEditMode ? "Update Pipeline Configuration" : "Create Pipeline"}
 message={
 <>
 {isEditMode ? (
 <>
 You are about to update the pipeline configuration with:
 <ul className="list-disc ml-5 mt-2 mb-2">
 <li><strong>{datasets.filter(d => d.selected).length}</strong> projects</li>
 <li><strong>{calculateTotalRuns()}</strong> total runs</li>
 <li><strong>~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</strong> estimated duration</li>
 </ul>
 <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-xs text-blue-800">
 <strong> Only increasing exploration runs?</strong> The pipeline will be set to <em>stopped</em> automatically click <strong>Resume</strong> to run the additional runs without losing previous results.
 </div>
 <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
 <strong> Changed projects, phases structure, or other settings?</strong> Those changes require a full <strong>Restart</strong> to take effect.
 </div>
 </>
 ) : (
 <>
 You are about to create a {pipelineType === "test" ? "testing" : "offline preparation"} pipeline with:
 <ul className="list-disc ml-5 mt-2 mb-2">
 <li><strong>{datasets.filter(d => d.selected).length}</strong> projects</li>
 <li><strong>{calculateTotalRuns()}</strong> total runs</li>
 <li><strong>~{calculateEstimatedTime().hours}h {calculateEstimatedTime().minutes}m</strong> estimated duration</li>
 </ul>
 The pipeline will be created with status "pending". You can start it manually from the pipeline list.
 <br /><br />
 Do you want to continue?
 </>
 )}
 </>
 }
 confirmLabel={isEditMode ? "Save Changes" : "Create Pipeline"}
 cancelLabel="Cancel"
 tone="default"
 busy={creating}
 onConfirm={handleCreatePipeline}
 onCancel={() => setShowCreateConfirm(false)}
 />

 {/* Toast Notification */}
 {toast && (
 <div
 style={{
 position: "fixed",
 bottom: "20px",
 right: "20px",
 padding: "12px 20px",
 background: toast.type === "success" ? "#4CAF50" : "#f44336",
 color: "white",
 borderRadius: "4px",
 boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
 zIndex: 9999,
 maxWidth: "400px",
 }}
 >
 {toast.message}
 </div>
 )}
 </main>
 </div>
 );
}

