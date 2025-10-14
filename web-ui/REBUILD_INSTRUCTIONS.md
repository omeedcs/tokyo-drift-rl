# Clean Research Website Rebuild

## Changes Made

### 1. Clean Black & White Theme
- Removed all neon/cyberpunk colors
- Professional grayscale palette
- Clean typography with Inter font
- Subtle shadows and borders

### 2. Research Paper Structure  
The new website follows academic paper format:

**Sections:**
- Hero: Project title, authors, abstract
- Problem Formulation: What we're solving
- Mathematical Framework: IKD & SAC explained with LaTeX
- Environment Design: Technical setup
- Results & Analysis: Performance metrics
- Interactive Demo: Embedded simulation
- Implementation Details: Code & architecture
- References: Papers and resources

### 3. Dependencies to Install
```bash
cd web-ui
npm install @tailwindcss/typography katex react-katex
```

### 4. Key Files to Create

Create these new files in `web-ui/src/`:

**app/page.tsx** - Main research page with all sections
**components/MathSection.tsx** - LaTeX equation renderer
**components/Hero.tsx** - Clean hero section
**components/DemoSection.tsx** - Embedded simulation
**components/Smooth3DViewer.tsx** - Fixed 3D (60fps, no glitches)

### 5. Run

```bash
cd web-ui
npm install
npm run dev
```

The site will be clean, professional, and explain everything in depth!
