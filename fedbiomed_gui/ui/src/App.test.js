import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import SideNav from './components/layout/SideNav';

test('renders node lifecycle navigation item', () => {
  render(
    <MemoryRouter>
      <SideNav />
    </MemoryRouter>
  );
  expect(screen.getByText(/Node Lifecycle/i)).toBeInTheDocument();
});
