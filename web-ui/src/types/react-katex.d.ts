declare module 'react-katex' {
  import { Component } from 'react'

  export interface KatexProps {
    math?: string
    children?: string
    errorColor?: string
    renderError?: (error: Error) => JSX.Element
  }

  export class InlineMath extends Component<KatexProps> {}
  export class BlockMath extends Component<KatexProps> {}
}
