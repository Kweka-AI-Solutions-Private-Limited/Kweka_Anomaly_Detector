import React, { useState } from 'react';
import { LucideIcon, Info } from 'lucide-react';
import { Card } from './Card';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: string;
  isPositive?: boolean;
  icon: LucideIcon;
  definition?: string;
  formula?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  isPositive = true,
  icon: Icon,
  definition,
  formula,
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <Card className="p-6 hover:border-brand-300 transition-all relative">
      <div className="flex items-center justify-between">
        <div
          className="flex items-center space-x-1.5 cursor-help group"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          <span className="text-xs font-mono font-black uppercase tracking-wider text-industrial-900 group-hover:text-brand-600 transition-colors">
            {title}
          </span>
          {(definition || formula) && (
            <Info className="w-3.5 h-3.5 text-industrial-400 group-hover:text-brand-600 transition-colors" />
          )}
        </div>

        <div className="p-3 rounded-xl bg-brand-50 border border-brand-200 text-brand-600">
          <Icon className="w-6 h-6" />
        </div>
      </div>

      <div className="mt-4">
        <span className="text-4xl font-black text-industrial-900 tracking-tight font-sans block">
          {value}
        </span>
        {change && (
          <p
            className={`text-sm font-mono mt-1.5 font-bold ${
              isPositive ? 'text-pass-700' : 'text-industrial-600'
            }`}
          >
            {change}
          </p>
        )}
      </div>

      {/* Hover Tooltip Popover */}
      {showTooltip && (definition || formula) && (
        <div className="absolute z-30 left-4 right-4 top-14 bg-industrial-900 text-white text-xs font-sans p-3 rounded-xl shadow-2xl border border-industrial-700 pointer-events-none space-y-1.5 animate-in fade-in duration-200">
          <div className="font-mono font-bold text-brand-300 uppercase tracking-wider text-[10px] border-b border-industrial-800 pb-1">
            {title} Metric Info
          </div>
          {definition && (
            <div>
              <span className="font-mono font-bold text-industrial-400 text-[10px] block uppercase">Definition:</span>
              <span className="text-industrial-200 text-[11px] leading-tight block">{definition}</span>
            </div>
          )}
          {formula && (
            <div className="bg-industrial-950 p-2 rounded-lg border border-industrial-800 font-mono text-[11px] text-amber-300">
              <span className="font-bold text-industrial-400 text-[10px] block uppercase mb-0.5">Formula:</span>
              <code>{formula}</code>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
