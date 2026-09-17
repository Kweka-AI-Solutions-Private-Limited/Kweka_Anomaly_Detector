import React from 'react';

interface LoadingSpinnerProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ label = 'Loading...', size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }[size];

  return (
    <div className="flex flex-col items-center justify-center p-8 text-center space-y-3">
      <div className={`${sizeClasses} border-3 border-industrial-300 border-t-brand-600 rounded-full animate-spin`} />
      {label && <p className="text-sm font-medium text-industrial-600 font-mono">{label}</p>}
    </div>
  );
};
