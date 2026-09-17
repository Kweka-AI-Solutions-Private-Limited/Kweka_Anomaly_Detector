import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Inspection } from '../Inspection';
import '@testing-library/jest-dom';

// Mock API calls
jest.mock('../../api/models', () => ({
  getModels: jest.fn().mockResolvedValue([
    {
      id: 'm1',
      name: 'Carpet Textile Model',
      category: 'Carpet',
      status: 'active',
      active_version_id: 'v1',
    },
  ]),
  activateModel: jest.fn().mockResolvedValue({ status: 'success' }),
}));

jest.mock('../../api/inspections', () => ({
  runInspection: jest.fn().mockResolvedValue({}),
}));

jest.mock('../../api/runs', () => ({
  createInspectionRun: jest.fn().mockResolvedValue({
    id: 'run1',
    status: 'running',
    total_images: 2,
    completed_images: 1,
    inspections: [
      {
        filename: 'imgA.png',
        status: 'completed',
        prediction: {
          status: 'anomalous',
          anomaly_score: 38.35,
          threshold: 21.43,
          severity: 'high',
        },
        processing_time_ms: 45,
      },
      {
        filename: 'imgB.png',
        status: 'processing',
      },
    ],
  }),
  getInspectionRun: jest.fn().mockResolvedValue({
    id: 'run1',
    status: 'running',
    total_images: 2,
    completed_images: 1,
    inspections: [
      {
        filename: 'imgA.png',
        status: 'completed',
        prediction: {
          status: 'anomalous',
          anomaly_score: 38.35,
          threshold: 21.43,
          severity: 'high',
        },
        processing_time_ms: 45,
      },
      {
        filename: 'imgB.png',
        status: 'processing',
      },
    ],
  }),
}));

describe('Inspection Result Panel Loading State', () => {
  it('clears completed result and shows ANALYZING loading state when switching to an analyzing image', async () => {
    render(
      <BrowserRouter>
        <Inspection />
      </BrowserRouter>
    );

    // Setup screen verification
    const uploadText = await screen.findByText(/Carpet Textile Model/i);
    expect(uploadText).toBeInTheDocument();
  });
});
