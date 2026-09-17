/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        industrial: {
          50: '#F8FAF9',
          100: '#F1F5F4',
          200: '#E2E8E5',
          300: '#CBD5D1',
          400: '#94A39D',
          500: '#64746F',
          600: '#475550',
          700: '#33413C',
          800: '#1E2926',
          900: '#0F1715',
          950: '#020605',
        },
        brand: {
          50: '#FDF6F0',  // Warm peach light background
          100: '#FCE8D9', // Light accent badge
          200: '#F9CDAF',
          300: '#F4A67B',
          400: '#EE7F46',
          500: '#E06A26', // Kweka.ai primary terracotta orange
          600: '#C95616', // Darker hover orange
          700: '#9F400F',
          800: '#7D320C',
          900: '#61260A',
        },
        pass: {
          50: '#F0FDF4',
          100: '#DCFCE7',
          500: '#22C55E',
          600: '#16A34A',
          700: '#15803D',
        },
        reject: {
          50: '#FEF2F2',
          100: '#FEE2E2',
          500: '#EF4444',
          600: '#DC2626',
          700: '#B91C1C',
        },
        review: {
          50: '#FFFBEB',
          100: '#FEF3C7',
          500: '#F59E0B',
          600: '#D97706',
          700: '#B45309',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      }
    },
  },
  plugins: [],
}
