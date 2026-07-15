import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { Boxes, FileText, GitBranch, Images, Play, Table2 } from "lucide-react";
import { api } from "../../api/client";
import ProjectPageHeader from "../../components/projectDetails/ProjectPageHeader";
import ProjectModelsReferencePanel from "../../components/projectDetails/ProjectModelsReferencePanel";
import ProjectTabsNav from "../../components/projectDetails/ProjectTabsNav";
import ImagesTab from "../../components/tabs/ImagesTab";
import LogsTab from "../../components/tabs/LogsTab";
import ProcessTab from "../../components/tabs/ProcessTab";
import ProjectTestResultsTab from "../../components/tabs/ProjectTestResultsTab";
import SessionsTab from "../../components/tabs/SessionsTab";

type TabType = "images" | "process" | "test_results" | "logs" | "sessions" | "models";

interface ProjectStatus {
  created_at?: string | null;
  name?: string | null;
  progress: number;
  project_id: string;
  status: string;
}

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const returnTo = searchParams.get("returnTo");
  const returnToPipeline = searchParams.get("returnToPipeline");
  const source = searchParams.get("from");
  const [activeTab, setActiveTab] = useState<TabType>("images");
  const [projectStatus, setProjectStatus] = useState<ProjectStatus | null>(null);
  const [hasImages, setHasImages] = useState(false);
  const [initialTabChosen, setInitialTabChosen] = useState(false);

  useEffect(() => {
    if (!id) return;

    const fetchStatus = async () => {
      try {
        const res = await api.get(`/projects/${id}/status`);
        setProjectStatus(res.data);

        const filesRes = await api.get(`/projects/${id}/files`);
        const has = filesRes.data.files?.images?.length > 0;
        setHasImages(has);
        if (!initialTabChosen) {
          setActiveTab(has ? "process" : "images");
          setInitialTabChosen(true);
        }

        if (res.data.status === "processing" && activeTab === "images") {
          setActiveTab("process");
        }
      } catch (err) {
        console.error("Failed to fetch project:", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [id, activeTab, initialTabChosen]);

  if (!id) return null;

  const tabs = [
    { id: "images" as TabType, label: "Images", icon: Images, enabled: true },
    { id: "process" as TabType, label: "Process", icon: Play, enabled: hasImages },
    { id: "test_results" as TabType, label: "Test Results", icon: Table2, enabled: true },
    { id: "logs" as TabType, label: "Logs", icon: FileText, enabled: true },
    { id: "sessions" as TabType, label: "Sessions", icon: Boxes, enabled: true },
    { id: "models" as TabType, label: "Models", icon: GitBranch, enabled: true },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <ProjectPageHeader
        projectId={id}
        projectStatus={projectStatus}
        returnTo={returnTo}
        returnToPipeline={returnToPipeline}
        source={source}
      />

      <ProjectTabsNav
        activeTab={activeTab}
        isProcessing={projectStatus?.status === "processing" || projectStatus?.status === "stopping"}
        onTabChange={setActiveTab}
        tabs={tabs}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
        {activeTab === "images" && (
          <ImagesTab
            projectId={id}
            onUploaded={() => {
              setHasImages(true);
              setActiveTab("process");
            }}
          />
        )}
        {activeTab === "process" && <ProcessTab projectId={id} />}
        {activeTab === "test_results" && <ProjectTestResultsTab projectId={id} />}
        {activeTab === "logs" && <LogsTab projectId={id} />}
        {activeTab === "sessions" && <SessionsTab projectId={id} />}
        {activeTab === "models" && <ProjectModelsReferencePanel projectId={id} />}
      </main>
    </div>
  );
}
