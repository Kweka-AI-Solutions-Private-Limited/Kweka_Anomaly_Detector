import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Eye, Filter, RefreshCw, ChevronLeft, ChevronRight, X, SlidersHorizontal } from 'lucide-react';
import { getInspections } from '../api/inspections';
import { getModels, getModelVersions } from '../api/models';
import { getInspectionRuns } from '../api/runs';
import { getStorageUrl } from '../api/client';
import { Inspection, Model, ModelVersion, InspectionRun } from '../types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

export const InspectionHistory: React.FC = () => {
  const navigate = useNavigate();

  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [inspectionRuns, setInspectionRuns] = useState<InspectionRun[]>([]);
  
  const [modelMap, setModelMap] = useState<Record<string, string>>({});
  const [versionMap, setVersionMap] = useState<Record<string, number>>({});
  
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filter Panel Toggle State
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState<boolean>(false);

  // Applied Filters State (drives data fetching & active chips)
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedVersion, setSelectedVersion] = useState<string>('all');
  const [selectedRun, setSelectedRun] = useState<string>('all');
  const [selectedVerdict, setSelectedVerdict] = useState<string>('all');
  const [selectedDefect, setSelectedDefect] = useState<string>('all');
  const [selectedLocation, setSelectedLocation] = useState<string>('all');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  // Draft Filters State inside Panel
  const [draftModel, setDraftModel] = useState<string>('all');
  const [draftVersion, setDraftVersion] = useState<string>('all');
  const [draftRun, setDraftRun] = useState<string>('all');
  const [draftVerdict, setDraftVerdict] = useState<string>('all');
  const [draftDefect, setDraftDefect] = useState<string>('all');
  const [draftLocation, setDraftLocation] = useState<string>('all');
  const [draftSeverity, setDraftSeverity] = useState<string>('all');
  const [draftStartDate, setDraftStartDate] = useState<string>('');
  const [draftEndDate, setDraftEndDate] = useState<string>('');

  // Pagination State
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(20);

  // Sync draft filters when opening panel
  useEffect(() => {
    if (isFilterPanelOpen) {
      setDraftModel(selectedModel);
      setDraftVersion(selectedVersion);
      setDraftRun(selectedRun);
      setDraftVerdict(selectedVerdict);
      setDraftDefect(selectedDefect);
      setDraftLocation(selectedLocation);
      setDraftSeverity(selectedSeverity);
      setDraftStartDate(startDate);
      setDraftEndDate(endDate);
    }
  }, [isFilterPanelOpen]);

  // Load Metadata (Models, Versions, Runs)
  useEffect(() => {
    async function loadMetadata() {
      try {
        const [modelsData, runsData] = await Promise.all([
          getModels(),
          getInspectionRuns().catch(() => []),
        ]);
        setModels(modelsData);
        setInspectionRuns(runsData);

        const mmap: Record<string, string> = {};
        modelsData.forEach((m) => {
          const id = m.id || m._id;
          if (id) mmap[id] = m.name;
        });
        setModelMap(mmap);

        // Fetch model versions across models
        const allVersions: ModelVersion[] = [];
        const vmap: Record<string, number> = {};

        await Promise.all(
          modelsData.map(async (m) => {
            const mId = m.id || m._id;
            if (!mId) return;
            try {
              const vList = await getModelVersions(mId);
              vList.forEach((v) => {
                allVersions.push(v);
                const vId = v.id || v._id;
                if (vId) vmap[vId] = v.version_number;
              });
            } catch {
              // ignore fetch error
            }
          })
        );
        setModelVersions(allVersions);
        setVersionMap(vmap);
      } catch (err: any) {
        console.error('Failed to load history metadata:', err);
      }
    }
    loadMetadata();
  }, []);

  // Fetch Inspections based on active applied filters
  const fetchInspectionsData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const params: Parameters<typeof getInspections>[0] = {};

      if (selectedModel !== 'all') params.model_id = selectedModel;
      if (selectedVersion !== 'all') params.model_version_id = selectedVersion;
      if (selectedRun !== 'all') params.run_id = selectedRun;
      if (selectedVerdict !== 'all') params.status = selectedVerdict;
      if (selectedDefect !== 'all') params.defect_type = selectedDefect;
      if (selectedLocation !== 'all') params.location = selectedLocation;
      if (selectedSeverity !== 'all') params.severity = selectedSeverity;
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      if (searchQuery.trim()) params.search = searchQuery.trim();

      const inspectionsData = await getInspections(params);
      setInspections(inspectionsData);
    } catch (err: any) {
      setError('Failed to fetch inspection history.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchInspectionsData();
  }, [
    selectedModel,
    selectedVersion,
    selectedRun,
    selectedVerdict,
    selectedDefect,
    selectedLocation,
    selectedSeverity,
    startDate,
    endDate,
    searchQuery,
  ]);

  // CASCADING FILTER DROPDOWN LOGIC:
  // 1. Available unique versions for selected draft model
  const availableVersions = useMemo(() => {
    if (draftModel === 'all') return [];
    const rawVersions = modelVersions.filter((v) => (v.model_id || (v as any)._id) === draftModel);
    
    // Deduplicate by version_number
    const uniqueMap = new Map<number, ModelVersion>();
    rawVersions.forEach((v) => {
      if (!uniqueMap.has(v.version_number)) {
        uniqueMap.set(v.version_number, v);
      }
    });
    return Array.from(uniqueMap.values()).sort((a, b) => a.version_number - b.version_number);
  }, [draftModel, modelVersions]);

  // 2. Available unique runs for selected draft model + draft version
  const availableRuns = useMemo(() => {
    if (draftModel === 'all' || draftVersion === 'all') return [];
    
    const rawRuns = inspectionRuns.filter((r) => {
      const mId = r.model_id || (r as any)._id;
      const vId = r.model_version_id || (r as any).model_version_id;
      return mId === draftModel && vId === draftVersion;
    });

    // Deduplicate by run_number
    const uniqueMap = new Map<number, InspectionRun>();
    rawRuns.forEach((r) => {
      if (!uniqueMap.has(r.run_number)) {
        uniqueMap.set(r.run_number, r);
      }
    });
    return Array.from(uniqueMap.values()).sort((a, b) => a.run_number - b.run_number);
  }, [draftModel, draftVersion, inspectionRuns]);

  // Handle Draft Model Change -> Resets Version & Run
  const handleDraftModelChange = (newModelId: string) => {
    setDraftModel(newModelId);
    setDraftVersion('all');
    setDraftRun('all');
  };

  // Handle Draft Version Change -> Resets Run
  const handleDraftVersionChange = (newVersionId: string) => {
    setDraftVersion(newVersionId);
    setDraftRun('all');
  };

  // Unique Defect Types & Locations extracted from inspection results
  const uniqueDefectTypes = useMemo(() => {
    const set = new Set<string>();
    inspections.forEach((item) => {
      const vlm = item.vlm_analysis || (item as any).result?.vlm_analysis;
      if (vlm?.defect_type && vlm.defect_type !== 'Not required' && vlm.defect_type !== 'None') {
        set.add(vlm.defect_type);
      }
    });
    return Array.from(set).sort();
  }, [inspections]);

  const uniqueLocations = useMemo(() => {
    const set = new Set<string>();
    inspections.forEach((item) => {
      const vlm = item.vlm_analysis || (item as any).result?.vlm_analysis;
      if (vlm?.location && vlm.location !== 'N/A') {
        set.add(vlm.location);
      }
    });
    return Array.from(set).sort();
  }, [inspections]);

  // Apply Filter Panel Changes
  const handleApplyFilters = () => {
    setSelectedModel(draftModel);
    setSelectedVersion(draftVersion);
    setSelectedRun(draftRun);
    setSelectedVerdict(draftVerdict);
    setSelectedDefect(draftDefect);
    setSelectedLocation(draftLocation);
    setSelectedSeverity(draftSeverity);
    setStartDate(draftStartDate);
    setEndDate(draftEndDate);
    setCurrentPage(1);
    setIsFilterPanelOpen(false);
  };

  // Clear Panel Draft Filters
  const handleClearPanelFilters = () => {
    setDraftModel('all');
    setDraftVersion('all');
    setDraftRun('all');
    setDraftVerdict('all');
    setDraftDefect('all');
    setDraftLocation('all');
    setDraftSeverity('all');
    setDraftStartDate('');
    setDraftEndDate('');
  };

  // Reset All Active Applied Filters
  const handleResetAllFilters = () => {
    setSearchQuery('');
    setSelectedModel('all');
    setSelectedVersion('all');
    setSelectedRun('all');
    setSelectedVerdict('all');
    setSelectedDefect('all');
    setSelectedLocation('all');
    setSelectedSeverity('all');
    setStartDate('');
    setEndDate('');
    setCurrentPage(1);
  };

  // Compute Active Filter Count & Active Chips List
  const activeFilters = useMemo(() => {
    const list: { key: string; label: string; clear: () => void }[] = [];

    if (selectedModel !== 'all') {
      const mName = modelMap[selectedModel] || 'Selected Model';
      list.push({ key: 'model', label: `Model: ${mName}`, clear: () => { setSelectedModel('all'); setSelectedVersion('all'); setSelectedRun('all'); } });
    }
    if (selectedVersion !== 'all') {
      const vNum = versionMap[selectedVersion];
      list.push({ key: 'version', label: `Version: V${vNum ?? selectedVersion.slice(-4)}`, clear: () => { setSelectedVersion('all'); setSelectedRun('all'); } });
    }
    if (selectedRun !== 'all') {
      if (selectedRun === 'standalone') {
        list.push({ key: 'run', label: 'Run: Standalone', clear: () => setSelectedRun('all') });
      } else {
        const rObj = inspectionRuns.find(r => (r.id || (r as any)._id) === selectedRun);
        list.push({ key: 'run', label: `Run: #${rObj?.run_number ?? selectedRun.slice(-4)}`, clear: () => setSelectedRun('all') });
      }
    }
    if (selectedVerdict !== 'all') {
      list.push({ key: 'verdict', label: `Verdict: ${selectedVerdict.toUpperCase()}`, clear: () => setSelectedVerdict('all') });
    }
    if (selectedDefect !== 'all') {
      list.push({ key: 'defect', label: `Defect: ${selectedDefect}`, clear: () => setSelectedDefect('all') });
    }
    if (selectedLocation !== 'all') {
      list.push({ key: 'location', label: `Location: ${selectedLocation}`, clear: () => setSelectedLocation('all') });
    }
    if (selectedSeverity !== 'all') {
      list.push({ key: 'severity', label: `Severity: ${selectedSeverity}`, clear: () => setSelectedSeverity('all') });
    }
    if (startDate) {
      list.push({ key: 'start_date', label: `From: ${startDate}`, clear: () => setStartDate('') });
    }
    if (endDate) {
      list.push({ key: 'end_date', label: `To: ${endDate}`, clear: () => setEndDate('') });
    }

    return list;
  }, [selectedModel, selectedVersion, selectedRun, selectedVerdict, selectedDefect, selectedLocation, selectedSeverity, startDate, endDate, modelMap, versionMap, inspectionRuns]);

  // Pagination calculation
  const totalRecords = inspections.length;
  const totalPages = Math.ceil(totalRecords / pageSize) || 1;
  const paginatedInspections = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return inspections.slice(start, start + pageSize);
  }, [inspections, currentPage, pageSize]);

  if (isLoading && inspections.length === 0) {
    return <LoadingSpinner label="Loading inspection history..." size="lg" />;
  }

  return (
    <div className="space-y-6">
      {/* Header & Clean Control Bar */}
      <div className="bg-white p-6 sm:p-7 rounded-2xl border border-industrial-200 shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-industrial-900 tracking-tight">Inspection History</h1>
          <p className="text-sm text-industrial-600 mt-1 font-normal">
            Individual inspection audit log across Model → Version → Run hierarchy.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3 self-start sm:self-auto">
          {/* Filters Toggle Button */}
          <button
            type="button"
            onClick={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
            className={`inline-flex items-center space-x-2 px-4 py-2.5 text-sm font-semibold rounded-xl border transition-colors shadow-sm ${
              activeFilters.length > 0 || isFilterPanelOpen
                ? 'bg-brand-50 border-brand-300 text-brand-700 font-bold'
                : 'bg-white border-industrial-300 text-industrial-700 hover:bg-industrial-50'
            }`}
          >
            <SlidersHorizontal className="w-4 h-4 text-brand-600" />
            <span>Filters</span>
            {activeFilters.length > 0 && (
              <span className="ml-1.5 px-2 py-0.5 text-xs font-bold font-mono bg-brand-600 text-white rounded-full">
                {activeFilters.length}
              </span>
            )}
          </button>

          {/* Reset / Clear Button */}
          {(activeFilters.length > 0 || searchQuery.trim()) && (
            <button
              type="button"
              onClick={handleResetAllFilters}
              className="inline-flex items-center space-x-1.5 px-3.5 py-2.5 text-sm font-semibold border border-industrial-300 rounded-xl hover:bg-industrial-50 text-industrial-600 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-reject-50 border border-reject-200 text-reject-700 text-sm rounded-xl font-semibold">
          {error}
        </div>
      )}

      {/* Primary Search Bar */}
      <div className="relative">
        <Search className="w-5 h-5 text-industrial-400 absolute left-4 top-3.5" />
        <input
          type="text"
          placeholder="Search by filename or model name..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            setCurrentPage(1);
          }}
          className="w-full pl-12 pr-4 py-3 text-sm sm:text-base border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium shadow-sm"
        />
      </div>

      {/* Active Filter Chips */}
      {activeFilters.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 bg-industrial-50 p-3.5 rounded-xl border border-industrial-200">
          <span className="text-xs font-bold text-industrial-500 uppercase font-mono mr-1">
            Active Filters ({activeFilters.length}):
          </span>
          {activeFilters.map((chip) => (
            <span
              key={chip.key}
              className="inline-flex items-center space-x-1 px-3 py-1 bg-white border border-brand-200 text-brand-800 text-xs font-semibold rounded-lg shadow-2xs font-mono"
            >
              <span>{chip.label}</span>
              <button
                type="button"
                onClick={chip.clear}
                className="text-brand-500 hover:text-brand-800 hover:bg-brand-50 rounded-full p-0.5 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </span>
          ))}
          <button
            type="button"
            onClick={handleResetAllFilters}
            className="text-xs font-bold text-brand-600 hover:underline ml-2 font-mono"
          >
            Clear All
          </button>
        </div>
      )}

      {/* Collapsible Filter Panel Modal / Drawer */}
      {isFilterPanelOpen && (
        <Card className="p-6 bg-white border border-brand-200 shadow-md space-y-5 animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-industrial-100 pb-3">
            <div className="flex items-center space-x-2 text-industrial-800 font-extrabold text-sm uppercase tracking-wider font-mono">
              <Filter className="w-4 h-4 text-brand-600" />
              <span>Filter Audit Trail</span>
            </div>
            <button
              type="button"
              onClick={() => setIsFilterPanelOpen(false)}
              className="text-industrial-400 hover:text-industrial-700 p-1 rounded-lg hover:bg-industrial-100 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Cascading Filter Controls Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* 1. Model Filter */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Model</label>
              <select
                value={draftModel}
                onChange={(e) => handleDraftModelChange(e.target.value)}
                className="w-full px-3.5 py-2.5 text-sm border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-semibold text-industrial-800"
              >
                <option value="all">All Models ({models.length})</option>
                {models.map((m) => (
                  <option key={m.id || m._id} value={m.id || m._id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* 2. Model Version Filter (Disabled until Model is selected) */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Version</label>
              <select
                value={draftVersion}
                disabled={draftModel === 'all'}
                onChange={(e) => handleDraftVersionChange(e.target.value)}
                className="w-full px-3.5 py-2.5 text-sm border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-semibold text-industrial-800 disabled:bg-industrial-100 disabled:text-industrial-400 disabled:cursor-not-allowed"
              >
                {draftModel === 'all' ? (
                  <option value="all">Select a Model first</option>
                ) : (
                  <>
                    <option value="all">All Versions ({availableVersions.length})</option>
                    {availableVersions.map((v) => {
                      const vId = v.id || v._id;
                      return (
                        <option key={vId} value={vId}>
                          Version {v.version_number}
                        </option>
                      );
                    })}
                  </>
                )}
              </select>
            </div>

            {/* 3. Inspection Run Filter (Disabled until Version is selected) */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Inspection Run</label>
              <select
                value={draftRun}
                disabled={draftModel === 'all' || draftVersion === 'all'}
                onChange={(e) => setDraftRun(e.target.value)}
                className="w-full px-3.5 py-2.5 text-sm border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-semibold text-industrial-800 disabled:bg-industrial-100 disabled:text-industrial-400 disabled:cursor-not-allowed"
              >
                {draftModel === 'all' || draftVersion === 'all' ? (
                  <option value="all">Select a Version first</option>
                ) : (
                  <>
                    <option value="all">All Runs ({availableRuns.length})</option>
                    <option value="standalone">Standalone / Legacy (No Run ID)</option>
                    {availableRuns.map((r) => {
                      const rId = r.id || r._id || r.run_id;
                      return (
                        <option key={rId} value={rId}>
                          Run #{r.run_number} ({r.total_images} images)
                        </option>
                      );
                    })}
                  </>
                )}
              </select>
            </div>
          </div>

          {/* Metadata & Date Range Filters Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 pt-3 border-t border-industrial-100">
            {/* Verdict */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Verdict</label>
              <select
                value={draftVerdict}
                onChange={(e) => setDraftVerdict(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              >
                <option value="all">All Verdicts</option>
                <option value="pass">PASS (Normal)</option>
                <option value="reject">REJECT (Anomalous)</option>
              </select>
            </div>

            {/* Defect Type */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Defect Type</label>
              <select
                value={draftDefect}
                onChange={(e) => setDraftDefect(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              >
                <option value="all">All Defect Types</option>
                {uniqueDefectTypes.map((dt) => (
                  <option key={dt} value={dt}>
                    {dt}
                  </option>
                ))}
              </select>
            </div>

            {/* Location */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Location</label>
              <select
                value={draftLocation}
                onChange={(e) => setDraftLocation(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              >
                <option value="all">All Locations</option>
                {uniqueLocations.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc}
                  </option>
                ))}
              </select>
            </div>

            {/* Severity */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Severity</label>
              <select
                value={draftSeverity}
                onChange={(e) => setDraftSeverity(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              >
                <option value="all">All Severities</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">Start Date</label>
              <input
                type="date"
                value={draftStartDate}
                onChange={(e) => setDraftStartDate(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-xs font-bold text-industrial-600 uppercase mb-1 font-mono">End Date</label>
              <input
                type="date"
                value={draftEndDate}
                onChange={(e) => setDraftEndDate(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-industrial-300 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 outline-none bg-white font-medium text-industrial-800"
              />
            </div>
          </div>

          {/* Panel Footer Action Buttons */}
          <div className="flex items-center justify-end space-x-3 pt-3 border-t border-industrial-100">
            <button
              type="button"
              onClick={handleClearPanelFilters}
              className="px-4 py-2 text-xs font-semibold text-industrial-600 hover:text-industrial-900 transition-colors"
            >
              Clear Panel
            </button>
            <button
              type="button"
              onClick={handleApplyFilters}
              className="px-5 py-2.5 text-xs font-bold bg-brand-600 hover:bg-brand-700 text-white rounded-xl shadow-sm transition-colors font-mono"
            >
              Apply Filters
            </button>
          </div>
        </Card>
      )}

      {/* History Audit Log Data Table */}
      <Card className="p-6 sm:p-7 space-y-5 bg-white border-industrial-200 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-sm sm:text-base font-mono font-bold uppercase tracking-wider text-industrial-800">
            Inspection Log ({totalRecords} Records)
          </h2>

          {/* Page size selector */}
          <div className="flex items-center space-x-2 text-xs font-mono text-industrial-600">
            <span>Per page:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCurrentPage(1);
              }}
              className="px-2 py-1 border border-industrial-300 rounded-lg bg-white font-bold"
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </div>

        {paginatedInspections.length === 0 ? (
          <div className="p-12 text-center text-industrial-400 font-mono text-sm border border-industrial-200 rounded-xl">
            No matching inspection records found.
          </div>
        ) : (
          <div className="overflow-x-auto border border-industrial-200 rounded-xl shadow-2xs">
            <table className="w-full text-left">
              <thead className="bg-industrial-50 text-industrial-700 font-mono border-b border-industrial-200 uppercase text-xs">
                <tr>
                  <th className="py-3.5 px-4">Thumbnail</th>
                  <th className="py-3.5 px-4">Filename</th>
                  <th className="py-3.5 px-4 font-bold text-industrial-900">Model</th>
                  <th className="py-3.5 px-4">Version</th>
                  <th className="py-3.5 px-4">Run</th>
                  <th className="py-3.5 px-4">Verdict</th>
                  <th className="py-3.5 px-4">Defect Type</th>
                  <th className="py-3.5 px-4">Severity</th>
                  <th className="py-3.5 px-4 font-mono">Score</th>
                  <th className="py-3.5 px-4">Timestamp</th>
                  <th className="py-3.5 px-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-industrial-200 bg-white font-mono text-xs sm:text-sm">
                {paginatedInspections.map((item) => {
                  const inspId = item.id || item.inspection_id || item._id;
                  const modelName = (item as any).model_name || modelMap[item.model_id] || 'Inspection Model';
                  const storageUrl = getStorageUrl(item.storage_uri || item.input?.storage_uri);
                  const pred = item.prediction || (item as any).result?.prediction;
                  const vlm = item.vlm_analysis || (item as any).result?.vlm_analysis;
                  
                  const verNum = versionMap[item.model_version_id];
                  const runNum = item.run_number || (item as any).run_number;
                  const runId = item.run_id || (item as any).run_id;

                  const predStatus = (pred?.status || 'normal').toLowerCase();
                  const isPass = predStatus === 'normal' || predStatus === 'pass';

                  // VLM metadata extraction
                  const defectType = isPass ? 'N/A' : (vlm?.defect_type || 'N/A');
                  const severityVal = isPass ? 'N/A' : (vlm?.severity || pred?.severity || 'N/A');

                  return (
                    <tr
                      key={inspId}
                      className="hover:bg-brand-50/40 cursor-pointer transition-colors"
                      onClick={() => navigate(`/inspections/${inspId}`)}
                    >
                      <td className="py-3 px-4">
                        {storageUrl ? (
                          <img
                            src={storageUrl}
                            alt="Sample"
                            className="w-10 h-10 object-cover rounded-lg border border-industrial-200 bg-industrial-100 shadow-2xs"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded-lg bg-industrial-100 border border-industrial-200 flex items-center justify-center text-xs text-industrial-400 font-sans">
                            N/A
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4 font-semibold text-industrial-900 truncate max-w-[150px]">
                        {item.filename || item.input?.filename || 'sample.png'}
                      </td>
                      <td className="py-3 px-4 font-sans font-extrabold text-industrial-800 truncate max-w-[150px]">
                        {modelName}
                      </td>
                      <td className="py-3 px-4 text-industrial-700">
                        {verNum != null ? `V${verNum}` : `V:${item.model_version_id?.slice(-4)}`}
                      </td>
                      <td className="py-3 px-4">
                        {runNum != null ? (
                          <span className="font-extrabold text-brand-700 font-mono">Run #{runNum}</span>
                        ) : runId ? (
                          <span className="font-extrabold text-brand-700 font-mono">Run #{runId.slice(-4)}</span>
                        ) : (
                          <span className="text-industrial-400 italic font-sans text-xs">Standalone</span>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        <Badge status={pred?.status || 'normal'} size="sm" />
                      </td>
                      <td className="py-3 px-4 font-sans font-medium text-industrial-700 max-w-[130px] truncate">
                        {defectType}
                      </td>
                      <td className="py-3 px-4 font-sans text-xs font-semibold text-industrial-700">
                        {severityVal}
                      </td>
                      <td className="py-3 px-4 text-industrial-800 font-bold">
                        {pred?.anomaly_score != null ? pred.anomaly_score.toFixed(2) : 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-industrial-500 text-xs whitespace-nowrap">
                        {item.created_at ? new Date(item.created_at).toLocaleString() : 'Just now'}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/inspections/${inspId}`);
                          }}
                          className="text-brand-600 hover:text-brand-700 hover:underline inline-flex items-center space-x-1 font-sans font-bold text-xs"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>View</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="flex flex-col sm:flex-row items-center justify-between pt-4 border-t border-industrial-200 gap-3">
            <span className="text-xs font-mono text-industrial-600">
              Page <strong className="text-industrial-900">{currentPage}</strong> of <strong className="text-industrial-900">{totalPages}</strong> ({totalRecords} items)
            </span>
            <div className="flex items-center space-x-2">
              <button
                type="button"
                disabled={currentPage === 1}
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1.5 text-xs font-semibold border border-industrial-300 rounded-lg hover:bg-industrial-50 disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-1 font-mono text-industrial-700"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>
              <span className="px-3 py-1 text-xs font-mono font-bold text-industrial-800 bg-industrial-100 rounded-lg">
                {currentPage}
              </span>
              <button
                type="button"
                disabled={currentPage === totalPages}
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                className="px-3 py-1.5 text-xs font-semibold border border-industrial-300 rounded-lg hover:bg-industrial-50 disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-1 font-mono text-industrial-700"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
