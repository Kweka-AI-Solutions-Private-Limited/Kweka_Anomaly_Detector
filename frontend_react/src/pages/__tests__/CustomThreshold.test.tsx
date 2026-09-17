import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ModelDetails } from '../ModelDetails';
import '@testing-library/jest-dom';

jest.mock('../../api/models', () => ({
  getModel: jest.fn().mockResolvedValue({
    id: 'model123',
    name: 'Tile Surface Anomaly Model',
    description: 'PatchCore Anomaly Detector for Tile Surface Inspection',
    domain: 'tile',
    status: 'active',
    active_version_id: 'ver1',
    reference_image_count: 5,
    created_at: '2026-09-11T12:00:00Z',
    updated_at: '2026-09-11T12:00:00Z',
  }),
  getModelVersions: jest.fn().mockResolvedValue([
    {
      id: 'ver1',
      version_number: 1,
      status: 'ready',
      calibration: {
        method: '95th_percentile',
        threshold: 21.43,
        auto_calibrated_threshold: 21.43,
        is_custom: false,
      },
      training: {
        build_time_ms: 150,
        reference_count: 5,
      },
      created_at: '2026-09-11T12:00:00Z',
    },
  ]),
  updateModelThreshold: jest.fn().mockResolvedValue({
    id: 'ver2',
    version_number: 2,
    status: 'ready',
    calibration: {
      method: 'custom',
      threshold: 18.5,
      auto_calibrated_threshold: 21.43,
      is_custom: true,
    },
    created_at: '2026-09-11T12:10:00Z',
  }),
}));

jest.mock('../../api/runs', () => ({
  getInspectionRuns: jest.fn().mockResolvedValue([]),
  getInspectionRun: jest.fn().mockResolvedValue(null),
}));

jest.mock('../../api/inspections', () => ({
  getInspections: jest.fn().mockResolvedValue([]),
}));

describe('Custom Threshold UI Component', () => {
  it('renders Adjust Anomaly Threshold card with automatic calibration status', async () => {
    render(
      <MemoryRouter initialEntries={['/models/model123']}>
        <Routes>
          <Route path="/models/:modelId" element={<ModelDetails />} />
        </Routes>
      </MemoryRouter>
    );

    const title = await screen.findByText(/Adjust Anomaly Threshold/i);
    expect(title).toBeInTheDocument();

    expect(screen.getByText(/AUTOMATIC \(95th Percentile\)/i)).toBeInTheDocument();
    expect(screen.getByText('21.43')).toBeInTheDocument();
  });

  it('allows user to enter custom threshold and calls updateModelThreshold', async () => {
    const { updateModelThreshold } = require('../../api/models');

    render(
      <MemoryRouter initialEntries={['/models/model123']}>
        <Routes>
          <Route path="/models/:modelId" element={<ModelDetails />} />
        </Routes>
      </MemoryRouter>
    );

    await screen.findByText(/Adjust Anomaly Threshold/i);

    const input = screen.getByPlaceholderText(/e.g. 20.0/i);
    fireEvent.change(input, { target: { value: '18.5' } });

    const submitBtn = screen.getByRole('button', { name: /Save Threshold/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(updateModelThreshold).toHaveBeenCalledWith('model123', 18.5);
    });
  });
});
