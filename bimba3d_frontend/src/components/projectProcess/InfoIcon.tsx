import type { ComponentProps } from "react";
import { Info as LucideInfo } from "lucide-react";

export const InfoIcon = (props: ComponentProps<typeof LucideInfo>) => (
  <LucideInfo className={props.className ? `${props.className} w-3 h-3` : "w-3 h-3"} {...props} />
);
