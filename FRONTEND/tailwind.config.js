/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        poppins: ["Poppins", "sans-serif"],
        rubikVinyl: ['"Rubik Vinyl"', "cursive"],
      },

      clipPath: {
        parallelogram:
          "polygon(50px 0, 100% 0, calc(100% - 50px) 100%, 0 100%)",
      },
      colors: {
        color: {
          1: "#AC6AFF",
          2: "#FFC876",
          3: "#FF776F",
          4: "#7ADB78",
          5: "#858DFF",
          6: "#FF98E2",
        },
        stroke: {
          1: "#26242C",
        },
        n: {
          1: "#FFFFFF",
          2: "#CAC6DD",
          3: "#ADA8C3",
          4: "#757185",
          5: "#3F3A52",
          6: "#252134",
          7: "#15131D",
          8: "#0E0C15",
          9: "#474060",
          10: "#43435C",
          11: "#1B1B2E",
          12: "#2E2A41",
          13: "#6C7275",
        },
      },

      spacing: {
        0.25: "0.0625rem",
        7.5: "1.875rem",
        15: "3.75rem",
      },
      opacity: {
        15: ".15",
      },
      transitionDuration: {
        DEFAULT: "200ms",
      },
      transitionTimingFunction: {
        DEFAULT: "linear",
      },
      zIndex: {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
      },
      borderWidth: {
        DEFAULT: "0.0625rem",
      },
      //   colors: {
      //     "mine-shaft": {
      //       50: "#f6f6f6",
      //       100: "#e7e7e7",
      //       200: "#d1d1d1",
      //       300: "#b0b0b0",
      //       400: "#888888",
      //       500: "#6d6d6d",
      //       600: "#5d5d5d",
      //       700: "#4f4f4f",
      //       800: "#454545",
      //       900: "#3d3d3d",
      //       950: "#0d1b2a",
      //     },

      //     "bright-sun": {
      //       50: "#fffbeb",
      //       100: "#fff3c6",
      //       200: "#ffe588",
      //       300: "#ffd148",
      //       400: "#ffbd20",
      //       500: "#f99b07",
      //       600: "#dd7302",
      //       700: "#b75006",
      //       800: "#943d0c",
      //       900: "#7a330d",
      //       950: "#461902",
      //     },

      //     "light-seletor": {
      //       50: "#ffffff",
      //       100: "#f8f9fa",
      //       200: "#f9f7f3",
      //     },

      //     black: {
      //       50: "#f6f6f6",
      //       100: "#e7e7e7",
      //       200: "#d1d1d1",
      //       300: "#b0b0b0",
      //       400: "#888888",
      //       500: "#6d6d6d",
      //       600: "#5d5d5d",
      //       700: "#4f4f4f",
      //       800: "#454545",
      //       900: "#3d3d3d",
      //       950: "#000000",
      //     },
      //   },
    },
  },
  plugins: [],
};
