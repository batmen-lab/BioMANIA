import { useContext } from 'react';
import { useTranslation } from 'next-i18next';
import HomeContext from '@/pages/api/home/home.context';
import { libImages } from './LibCardSelect';

export const LibSelect = () => {
  const { t } = useTranslation('chat');

  const {
    state: { selectedConversation, methods, defaultMethodId },
    handleUpdateConversation,
    dispatch: homeDispatch,
  } = useContext(HomeContext);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (selectedConversation) {
      handleUpdateConversation(selectedConversation, {
        key: 'Lib',
        value: e.target.value,
      });
    }
  };

  const options = [];
  for (const libKey in libImages) {
    if (libImages.hasOwnProperty(libKey)) {
      options.push(
        <option
          key={libKey}
          value={libKey}
          className="dark:bg-[#343541] dark:text-white"
        >
          {libKey}
        </option>
      );
    }
  }

  return (
    <div className="flex flex-col">
      <label className="mb-2 text-left text-neutral-700 dark:text-neutral-400">
        {t('Lib')}
      </label>
      <div className="w-full rounded-lg border border-neutral-200 bg-transparent pr-2 text-neutral-900 dark:border-neutral-600 dark:text-white">
        <select
          className="w-full bg-transparent p-2"
          placeholder={t('Select a library') || ''}
          value={selectedConversation?.Lib || ''}
          onChange={handleChange}
        >
          {options}
        </select>
      </div>
    </div>
  );
};