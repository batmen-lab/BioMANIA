import { IconPlus } from '@tabler/icons-react';
import { FC } from 'react';
import { useTranslation } from 'next-i18next';

import { SidebarButton } from '../Sidebar/SidebarButton';

interface Props {
  onUpload: (data: any) => void;
}

export const UploadData: FC<Props> = ({ onUpload }) => {
  const { t } = useTranslation('sidebar');

  return (
    <>
      <input
        id="upload-file"
        className="sr-only"
        tabIndex={-1}
        type="file"
        // accept=".json"
        onChange={(e) => {
          if (!e.target.files?.length) return;

          const file = e.target.files[0];
          const reader = new FileReader();
          reader.onload = (e) => {
            const fileData = {
              name: file.name,
              size: file.size,
              type: file.type,
              content: e.target?.result
            };
            onUpload(fileData);
          };
          reader.readAsText(file);
        }}
      />

      <SidebarButton
        text={t('Upload data')}
        icon={<IconPlus size={18} />}
        onClick={() => {
          const uploadFile = document.querySelector('#upload-file') as HTMLInputElement;
          if (uploadFile) {
            uploadFile.click();
          }
        }}
      />
    </>
  );
};
