import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({ children, className = '', onClick }) => {
  return (
    <div
      onClick={onClick}
      className={`bg-white rounded-lg border border-industrial-200 shadow-sm overflow-hidden ${
        onClick ? 'cursor-pointer hover:border-industrial-300 transition-colors' : ''
      } ${className}`}
    >
      {children}
    </div>
  );
};
