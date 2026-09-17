import React from 'react';

interface BadgeProps {
  status?: string;
  type?: 'pass' | 'reject' | 'review' | 'ready' | 'draft' | 'info';
  children?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
}

export const Badge: React.FC<BadgeProps> = ({ status, type, children, size = 'md' }) => {
  const normStatus = (status || '').toLowerCase();
  
  let badgeStyle = 'bg-industrial-200 text-industrial-800 border-industrial-300';
  let label = children || status;

  if (type === 'pass' || normStatus === 'pass' || normStatus === 'normal' || normStatus === 'active') {
    badgeStyle = 'bg-emerald-100 text-emerald-800 border-emerald-300 font-extrabold';
    if (!children && (normStatus === 'normal' || normStatus === 'pass')) label = 'PASS';
    else if (!children && normStatus === 'active') label = 'ACTIVE';
  } else if (normStatus === 'inactive') {
    badgeStyle = 'bg-industrial-200 text-industrial-800 border-industrial-300 font-extrabold';
    if (!children) label = 'INACTIVE';
  } else if (type === 'reject' || normStatus === 'reject' || normStatus === 'anomalous' || normStatus === 'failed' || normStatus === 'error') {
    badgeStyle = 'bg-rose-100 text-rose-800 border-rose-300 font-extrabold';
    if (!children && (normStatus === 'anomalous' || normStatus === 'reject')) label = 'REJECT';
    else if (!children && normStatus === 'error') label = 'ERROR';
  } else if (type === 'review' || normStatus === 'building' || normStatus === 'processing' || normStatus === 'queued') {
    badgeStyle = 'bg-amber-100 text-amber-800 border-amber-300 font-extrabold animate-pulse';
  } else if (type === 'draft' || normStatus === 'draft') {
    badgeStyle = 'bg-industrial-200 text-industrial-800 border-industrial-300 font-extrabold';
  } else if (type === 'info' || normStatus === 'info') {
    badgeStyle = 'bg-brand-100 text-brand-800 border-brand-300 font-extrabold';
  }

  const sizeClasses = {
    sm: 'px-3 py-1 text-xs font-bold tracking-wide',
    md: 'px-3.5 py-1.5 text-sm font-extrabold tracking-wider',
    lg: 'px-4.5 py-2 text-base font-extrabold tracking-wider',
  }[size];

  return (
    <span
      className={`inline-flex items-center justify-center font-mono rounded-lg border ${sizeClasses} ${badgeStyle} uppercase shadow-sm`}
    >
      {label}
    </span>
  );
};
