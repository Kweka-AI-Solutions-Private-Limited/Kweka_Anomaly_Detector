import React, { useEffect, useState } from 'react';
import {
  X, Images, Calendar, HardDrive, Search, ChevronLeft, ChevronRight,
  Eye, CheckCircle2, AlertCircle
} from 'lucide-react';
import { ModelVersion, ReferenceImage } from '../../types';
import { getVersionReferenceImages } from '../../api/models';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface ReferenceImagesModalProps {
  modelId: string;
  version: ModelVersion;
  onClose: () => void;
}

const ITEMS_PER_PAGE = 12;

export const ReferenceImagesModal: React.FC<ReferenceImagesModalProps> = ({
  modelId,
  version,
  onClose,
}) => {
  const [references, setReferences] = useState<ReferenceImage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [selectedImage, setSelectedImage] = useState<ReferenceImage | null>(null);

  useEffect(() => {
    async function loadReferences() {
      try {
        setIsLoading(true);
        setError(null);
        const vId = version.id || version._id || '';
        const data = await getVersionReferenceImages(modelId, vId);
        setReferences(data);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to load version reference images.');
      } finally {
        setIsLoading(false);
      }
    }

    loadReferences();
  }, [modelId, version]);

  const filteredReferences = references.filter((ref) =>
    ref.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalPages = Math.ceil(filteredReferences.length / ITEMS_PER_PAGE) || 1;
  const paginatedReferences = filteredReferences.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileSize = (ref: ReferenceImage): number => {
    return ref.file_size ?? ref.file_size_bytes ?? 0;
  };

  const getUploadedDate = (ref: ReferenceImage): string => {
    return ref.uploaded_at ?? ref.created_at ?? '';
  };

  const getImageUrl = (uri?: string, relPath?: string): string => {
    const rawPath = uri || relPath || '';
    if (!rawPath) return '';
    const cleanPath = rawPath.replace(/^\//, '');
    return `http://localhost:8000/${cleanPath}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-industrial-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl border border-industrial-200 shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="p-6 bg-industrial-900 text-white flex items-center justify-between border-b border-industrial-800">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-brand-500/20 text-brand-400 rounded-xl border border-brand-500/30">
              <Images className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-3">
                <h2 className="text-xl font-extrabold tracking-tight text-white">
                  Version #{version.version_number} — Reference Images
                </h2>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800 flex items-center space-x-1">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>GOOD / REFERENCE</span>
                </span>
              </div>
              <p className="text-xs text-industrial-400 font-mono mt-1">
                {references.length} GOOD reference images used to construct Version #{version.version_number} memory bank
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-industrial-400 hover:text-white hover:bg-industrial-800 rounded-xl transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Controls Bar: Search & Counts */}
        <div className="p-4 bg-industrial-50 border-b border-industrial-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-industrial-400" />
            <input
              type="text"
              placeholder="Search reference by filename..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full pl-9 pr-4 py-2 text-sm bg-white border border-industrial-200 rounded-xl text-industrial-900 placeholder-industrial-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 font-mono"
            />
          </div>

          <div className="text-xs font-mono text-industrial-500 flex items-center space-x-4">
            <span>Showing {filteredReferences.length} of {references.length} reference images</span>
          </div>
        </div>

        {/* Modal Content Body */}
        <div className="p-6 overflow-y-auto flex-1 min-h-[350px]">
          {isLoading ? (
            <div className="py-16">
              <LoadingSpinner label="Loading version reference images..." size="lg" />
            </div>
          ) : error ? (
            <div className="p-6 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl flex items-center space-x-3 font-mono text-sm">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          ) : filteredReferences.length === 0 ? (
            <div className="py-16 text-center space-y-3">
              <Images className="w-12 h-12 text-industrial-300 mx-auto" />
              <h3 className="text-lg font-bold text-industrial-800">No Reference Images Found</h3>
              <p className="text-sm text-industrial-500 max-w-sm mx-auto">
                {searchQuery
                  ? `No reference image matched query "${searchQuery}".`
                  : `No reference images recorded for Version #${version.version_number}.`}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {paginatedReferences.map((ref) => {
                const imgUrl = getImageUrl(ref.storage?.uri, ref.relative_path);
                const rId = ref.id || ref._id;

                return (
                  <div
                    key={rId}
                    onClick={() => setSelectedImage(ref)}
                    className="group bg-white rounded-xl border border-industrial-200 overflow-hidden hover:border-brand-500 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between"
                  >
                    {/* Thumbnail Area */}
                    <div className="relative aspect-square bg-industrial-900 flex items-center justify-center overflow-hidden">
                      {imgUrl ? (
                        <img
                          src={imgUrl}
                          alt={ref.filename}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />
                      ) : (
                        <div className="text-industrial-500 flex flex-col items-center">
                          <Images className="w-8 h-8 mb-1" />
                          <span className="text-[10px] font-mono">No Image</span>
                        </div>
                      )}

                      {/* Overlay Badge */}
                      <div className="absolute top-2 left-2 px-2 py-0.5 bg-emerald-950/90 text-emerald-300 text-[10px] font-mono font-bold rounded-md border border-emerald-800">
                        GOOD
                      </div>

                      <div className="absolute inset-0 bg-industrial-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="px-3 py-1.5 bg-white/90 text-industrial-900 rounded-lg text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg">
                          <Eye className="w-3.5 h-3.5" />
                          <span>Inspect</span>
                        </span>
                      </div>
                    </div>

                    {/* Metadata Footer */}
                    <div className="p-3 space-y-1.5 border-t border-industrial-100 bg-white">
                      <p className="text-xs font-mono font-bold text-industrial-900 truncate" title={ref.filename}>
                        {ref.filename}
                      </p>

                      <div className="flex items-center justify-between text-[11px] font-mono text-industrial-400">
                        <span className="flex items-center space-x-1">
                          <HardDrive className="w-3 h-3" />
                          <span>{formatFileSize(getFileSize(ref))}</span>
                        </span>

                        <span className="flex items-center space-x-1">
                          <Calendar className="w-3 h-3" />
                          <span>
                            {getUploadedDate(ref) ? new Date(getUploadedDate(ref)).toLocaleDateString() : 'N/A'}
                          </span>
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Modal Footer & Pagination */}
        <div className="p-4 bg-industrial-50 border-t border-industrial-200 flex items-center justify-between">
          <div className="text-xs font-mono text-industrial-500">
            Page {currentPage} of {totalPages}
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              icon={<ChevronLeft className="w-4 h-4" />}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              icon={<ChevronRight className="w-4 h-4" />}
            >
              Next
            </Button>
          </div>
        </div>
      </div>

      {/* Lightbox Modal Overlay */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-60 bg-industrial-950/90 backdrop-blur-md flex items-center justify-center p-6 animate-in fade-in duration-150"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] bg-industrial-900 border border-industrial-700 rounded-2xl overflow-hidden shadow-2xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-4 bg-industrial-900 border-b border-industrial-800 flex items-center justify-between text-white font-mono text-sm">
              <div className="flex items-center space-x-3">
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                  GOOD REFERENCE
                </span>
                <span className="font-bold">{selectedImage.filename}</span>
              </div>
              <button
                onClick={() => setSelectedImage(null)}
                className="p-1 text-industrial-400 hover:text-white rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 bg-black flex items-center justify-center overflow-auto max-h-[70vh]">
              <img
                src={getImageUrl(selectedImage.storage?.uri, selectedImage.relative_path)}
                alt={selectedImage.filename}
                className="max-h-[65vh] w-auto object-contain rounded-lg shadow-md"
              />
            </div>

            <div className="p-4 bg-industrial-900 border-t border-industrial-800 grid grid-cols-3 text-xs font-mono text-industrial-400">
              <div>
                <span className="text-industrial-500 block">FILENAME</span>
                <span className="text-white font-bold">{selectedImage.filename}</span>
              </div>
              <div>
                <span className="text-industrial-500 block">FILE SIZE</span>
                <span className="text-white font-bold">{formatFileSize(getFileSize(selectedImage))}</span>
              </div>
              <div>
                <span className="text-industrial-500 block">UPLOADED DATE</span>
                <span className="text-white font-bold">
                  {getUploadedDate(selectedImage) ? new Date(getUploadedDate(selectedImage)).toLocaleString() : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
