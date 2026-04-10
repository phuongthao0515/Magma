import type { FC, ReactNode } from "react";
import { Layout, Typography } from "antd";
import styled from "styled-components";

const { Content } = Layout;
const { Title } = Typography;

const StyledContent = styled(Content)`
  padding: 24px;
  max-width: 960px;
  margin: 0 auto;
  width: 100%;
`;

interface ContentLayoutProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export const ContentLayout: FC<ContentLayoutProps> = ({
  title,
  subtitle,
  children,
}) => {
  return (
    <StyledContent>
      <Title level={3}>{title}</Title>
      {subtitle && (
        <Typography.Text type="secondary">{subtitle}</Typography.Text>
      )}
      <div className="mt-4">{children}</div>
    </StyledContent>
  );
};
